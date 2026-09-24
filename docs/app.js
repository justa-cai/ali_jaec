// In-browser inference for the ali_jaec front-end.
//
// Everything happens locally: the model is fetched from ./models/, the audio
// from ./audio/ or from the user's file picker. No data leaves the page.
//
// ONNX Runtime Web runs WebAssembly on a single thread on purpose. Multi-thread
// mode needs SharedArrayBuffer, which needs COOP/COEP response headers, which
// GitHub Pages cannot send. Single-threaded SIMD is fast enough here: roughly
// 2 s for 10 s of audio.

const SR = 16000;
const SEGMENT = 48000;          // must match --segment used in export_onnx.py
const MODEL_URL = 'models/aec_lp.onnx';
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.mjs';

let ortPromise = null;
let sessionPromise = null;

function getOrt() {
  if (!ortPromise) {
    ortPromise = import(ORT_URL).then((ort) => {
      ort.env.wasm.numThreads = 1;
      ort.env.logLevel = 'error';
      return ort;
    });
  }
  return ortPromise;
}

function getSession() {
  if (!sessionPromise) {
    sessionPromise = getOrt().then(async (ort) => {
      const buf = await (await fetch(MODEL_URL)).arrayBuffer();
      const t0 = performance.now();
      const sess = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
      return { ort, sess, loadMs: performance.now() - t0 };
    });
  }
  return sessionPromise;
}

// ------------------------------------------------------------------ WAV I/O
async function readWav(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`cannot fetch ${url}`);
  return parseWav(await res.arrayBuffer());
}

function parseWav(buf) {
  const dv = new DataView(buf);
  const tag = (o) => String.fromCharCode(dv.getUint8(o), dv.getUint8(o + 1),
                                         dv.getUint8(o + 2), dv.getUint8(o + 3));
  if (buf.byteLength < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') {
    throw new Error('not a RIFF/WAVE file');
  }
  let channels = 1, rate = SR, format = 1, bits = 16, offset = 0, length = 0;
  for (let p = 12; p + 8 <= buf.byteLength;) {
    const id = tag(p);
    const size = dv.getUint32(p + 4, true);
    if (id === 'fmt ') {
      format = dv.getUint16(p + 8, true);
      channels = dv.getUint16(p + 10, true);
      rate = dv.getUint32(p + 12, true);
      bits = dv.getUint16(p + 22, true);
    } else if (id === 'data') {
      offset = p + 8;
      length = Math.min(size, buf.byteLength - offset);
    }
    p += 8 + size + (size & 1);
  }
  if (!offset) throw new Error('WAVE file has no data chunk');
  if (format === 0xfffe) format = 1;

  const frames = Math.floor(length / (bits / 8) / channels);
  const out = [];
  for (let c = 0; c < channels; c++) out.push(new Float32Array(frames));

  if (format === 3 && bits === 32) {
    for (let i = 0; i < frames; i++)
      for (let c = 0; c < channels; c++)
        out[c][i] = dv.getFloat32(offset + 4 * (i * channels + c), true);
  } else if (format === 1 && bits === 16) {
    for (let i = 0; i < frames; i++)
      for (let c = 0; c < channels; c++)
        out[c][i] = dv.getInt16(offset + 2 * (i * channels + c), true) / 32768;
  } else {
    throw new Error(`unsupported WAV: format ${format}, ${bits} bit`);
  }
  return { channels, rate, data: out };
}

function encodeWav(samples, rate = SR) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + 2 * n);
  const dv = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); dv.setUint32(4, 36 + 2 * n, true); str(8, 'WAVE');
  str(12, 'fmt '); dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, rate, true); dv.setUint32(28, rate * 2, true);
  dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
  str(36, 'data'); dv.setUint32(40, 2 * n, true);
  for (let i = 0; i < n; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    dv.setInt16(44 + 2 * i, Math.round(v * 32767), true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

// -------------------------------------------------------------- processing
function resample(x, rateIn, rateOut = SR) {
  if (rateIn === rateOut) return x;
  const n = Math.round(x.length * rateOut / rateIn);
  const y = new Float32Array(n);
  const step = (x.length - 1) / Math.max(n - 1, 1);
  for (let i = 0; i < n; i++) {
    const p = i * step, i0 = Math.floor(p), i1 = Math.min(i0 + 1, x.length - 1);
    const t = p - i0;
    y[i] = x[i0] * (1 - t) + x[i1] * t;
  }
  return y;
}

function crossfadeWindow(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * (i + 1) / (n + 1));
  return w;
}

// The exported graph has static shapes, so long recordings are cut into
// SEGMENT-sized chunks with a 50 % overlap and raised-cosine weighted back
// together. That hides the fact that the recurrent network's context restarts
// at every boundary.
async function process(mic, ref, onProgress) {
  const { ort, sess, loadMs } = await getSession();
  const n = Math.min(mic.length, ref.length);
  const taus = [];

  const runSegment = async (micSeg, refSeg) => {
    const out = await sess.run({
      mic: new ort.Tensor('float32', micSeg, [1, SEGMENT]),
      ref: new ort.Tensor('float32', refSeg, [1, SEGMENT]),
    });
    taus.push(out.tau.data[0]);
    return out.out.data;
  };

  const t0 = performance.now();
  if (n <= SEGMENT) {
    const paddedM = new Float32Array(SEGMENT), paddedR = new Float32Array(SEGMENT);
    paddedM.set(mic.subarray(0, n)); paddedR.set(ref.subarray(0, n));
    const y = await runSegment(paddedM, paddedR);
    return { out: y.slice(0, n), tau: taus[0], ms: performance.now() - t0, loadMs, seq: 1 };
  }

  const hop = SEGMENT >> 1;
  const w = crossfadeWindow(SEGMENT);
  const num = new Float64Array(n), den = new Float64Array(n);
  const starts = [];
  for (let s = 0; s + SEGMENT < n; s += hop) starts.push(s);
  starts.push(n - SEGMENT);

  for (let k = 0; k < starts.length; k++) {
    const s0 = starts[k];
    const y = await runSegment(mic.slice(s0, s0 + SEGMENT),
                               ref.slice(s0, s0 + SEGMENT));
    for (let i = 0; i < SEGMENT; i++) {
      num[s0 + i] += w[i] * y[i];
      den[s0 + i] += w[i];
    }
    if (onProgress) onProgress((k + 1) / starts.length);
  }

  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = num[i] / Math.max(den[i], 1e-9);
  const sorted = taus.slice().sort((a, b) => a - b);
  return { out: y, tau: sorted[sorted.length >> 1], ms: performance.now() - t0,
           loadMs, seq: starts.length };
}

// ------------------------------------------------------------------ metrics
const db = (x) => 10 * Math.log10(x.reduce((a, v) => a + v * v, 0) / x.length + 1e-20);

// ERLE over 100 ms blocks whose far end is active, i.e. the same definition the
// Python evaluation uses. There is no clean near-end reference for this demo
// pair, so SI-SDR cannot be computed here -- ERLE alone is honest.
function erle(mic, out, ref, win = 1600) {
  const n = Math.min(mic.length, out.length, ref.length);
  const refDb = db(ref.subarray(0, n));
  let sum = 0, count = 0;
  for (let i = 0; i + win < n; i += win >> 1) {
    const m = mic.subarray(i, i + win), o = out.subarray(i, i + win), r = ref.subarray(i, i + win);
    if (db(r) < refDb - 35 || db(m) < -80) continue;
    sum += db(m) - db(o); count++;
  }
  return count ? sum / count : NaN;
}

function maxDiff(a, b) {
  const n = Math.min(a.length, b.length);
  let mx = 0;
  for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs(a[i] - b[i]));
  return mx;
}

// ---------------------------------------------------------------- waveform
function drawWave(canvas, samples, color) {
  const dpr = window.devicePixelRatio || 1;
  // Height comes from CSS (--wave-h, a clamp() on the viewport) so the
  // waveform grows with the display instead of staying 72 px forever. Read the
  // laid-out height rather than hard-coding it; assigning width/height only
  // resizes the backing bitmap and does not feed back into the CSS box.
  const w = canvas.clientWidth, h = canvas.clientHeight || 72;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const mid = h / 2;
  ctx.strokeStyle = '#232b35';
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();

  const step = samples.length / w;
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.9;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < w; x++) {
    const from = Math.floor(x * step), to = Math.min(samples.length, Math.floor((x + 1) * step));
    let lo = 1, hi = -1;
    for (let i = from; i < to; i++) {
      const v = samples[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (lo > hi) { lo = 0; hi = 0; }
    const y0 = mid - hi * mid * 1.7, y1 = mid - lo * mid * 1.7;
    ctx.moveTo(x + 0.5, y0);
    ctx.lineTo(x + 0.5, Math.max(y1, y0 + 0.6));
  }
  ctx.stroke();
}

// -------------------------------------------------------- spectrogram tiles
// Four signals, one picture each, all computed here -- the page ships no
// pre-rendered spectrograms and no FFT library. The analysis uses the model's
// own front-end parameters (see stft.py: NFFT 512, HOP 160, periodic
// sqrt-Hann, causal framing), so the tiles show what the network actually
// looks at rather than a loosely related picture.

const NFFT = 512;
const HOP = 160;
const FREQ = NFFT / 2 + 1;
const FMAX = SR / 2;

// One fixed dB window for all four tiles. Per-tile normalisation would make
// the rows incomparable, which is the one thing this figure exists to do.
const DB_LO = -85;
const DB_HI = 25;

// Viridis, as six anchors. The canvas LUT and the CSS legend bar are both
// built from this list, so the legend cannot drift away from the picture.
const CMAP = [[68, 1, 84], [65, 68, 135], [42, 120, 142],
              [34, 168, 132], [122, 209, 81], [253, 231, 37]];

const LUT = (() => {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const p = i / 255 * (CMAP.length - 1);
    const a = Math.min(CMAP.length - 2, Math.floor(p)), f = p - a;
    for (let c = 0; c < 3; c++) {
      lut[3 * i + c] = Math.round(CMAP[a][c] * (1 - f) + CMAP[a + 1][c] * f);
    }
  }
  return lut;
})();

// Iterative radix-2 FFT, in place. The bit-reversal permutation and the
// twiddles are built once and reused by all ~4000 frames; recomputing
// Math.cos per butterfly is the usual 5-10x tax on a naive version.
const LOG2N = Math.round(Math.log2(NFFT));
const REV = new Uint16Array(NFFT);
const TWID = new Float64Array(NFFT);
for (let i = 0; i < NFFT; i++) {
  let r = 0;
  for (let b = 0; b < LOG2N; b++) if (i & (1 << b)) r |= 1 << (LOG2N - 1 - b);
  REV[i] = r;
}
for (let k = 0; k < NFFT / 2; k++) {
  TWID[2 * k] = Math.cos(-2 * Math.PI * k / NFFT);
  TWID[2 * k + 1] = Math.sin(-2 * Math.PI * k / NFFT);
}

function fft(re, im) {
  for (let i = 0; i < NFFT; i++) {
    const j = REV[i];
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= NFFT; len <<= 1) {
    const half = len >> 1, step = NFFT / len;
    for (let i = 0; i < NFFT; i += len) {
      for (let k = 0; k < half; k++) {
        const wr = TWID[2 * k * step], wi = TWID[2 * k * step + 1];
        const a = i + k, b = a + half;
        const vr = re[b] * wr - im[b] * wi;
        const vi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] += vr;        im[a] += vi;
      }
    }
  }
}

// sqrt-Hann, periodic -- the same formula as stft.py's sqrt_hann().
const WIN = new Float32Array(NFFT);
for (let i = 0; i < NFFT; i++) {
  WIN[i] = Math.sqrt(0.5 * (1 - Math.cos(2 * Math.PI * i / NFFT)));
}

// stft.py zero-pads the tail so every input sample lands inside a frame;
// without it the last (L - NFFT) % HOP samples would get no column at all.
function frameCount(len) {
  const pad = (HOP - ((len - NFFT) % HOP)) % HOP;
  return Math.floor((len + pad - NFFT) / HOP) + 1;
}

// (frames, FREQ) magnitudes in dB, time-major.
function spectrogram(x, frames) {
  const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
  const out = new Float32Array(frames * FREQ);
  for (let t = 0; t < frames; t++) {
    const o = t * HOP;
    for (let i = 0; i < NFFT; i++) {
      const s = o + i;
      re[i] = s < x.length ? x[s] * WIN[i] : 0;
      im[i] = 0;
    }
    fft(re, im);
    for (let b = 0; b < FREQ; b++) {
      const v = re[b] * re[b] + im[b] * im[b];
      out[t * FREQ + b] = 20 * Math.log10(Math.sqrt(v) + 1e-12);
    }
  }
  return out;
}

// Drawn straight at device resolution, taking the max over each destination
// cell's source range. Averaging would wash the harmonics out, and letting
// drawImage do the scaling drops every peak its 2x2 bilinear kernel misses --
// which is what a 998-column spectrogram hits on a phone.
function drawTile(canvas, spec, frames) {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (!cw || !ch) return;
  const w = Math.round(cw * dpr), h = Math.round(ch * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
  }
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  const px = img.data;
  const sx = frames / w, sy = FREQ / h;
  const scale = 255 / (DB_HI - DB_LO);

  for (let y = 0; y < h; y++) {
    // image row 0 is the top, so walk the frequency axis backwards
    const f0 = Math.floor(y * sy);
    const f1 = Math.min(FREQ, Math.max(f0 + 1, Math.ceil((y + 1) * sy)));
    for (let x = 0; x < w; x++) {
      const t0 = Math.floor(x * sx);
      const t1 = Math.min(frames, Math.max(t0 + 1, Math.ceil((x + 1) * sx)));
      let m = -Infinity;
      for (let f = f0; f < f1; f++) {
        const b = FREQ - 1 - f;
        for (let t = t0; t < t1; t++) {
          const v = spec[t * FREQ + b];
          if (v > m) m = v;
        }
      }
      // Clamp before indexing: a negative index into a Uint8ClampedArray is a
      // silent no-op, which would leave the pixel fully transparent and let the
      // page background through instead of painting the bottom of the ramp.
      const q = (m - DB_LO) * scale;
      const i = q > 0 ? (q < 255 ? q | 0 : 255) : 0;
      const p = 4 * (y * w + x);
      px[p] = LUT[3 * i];
      px[p + 1] = LUT[3 * i + 1];
      px[p + 2] = LUT[3 * i + 2];
      px[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  // Frequency ticks, drawn after the image: putImageData ignores the canvas
  // transform, so the text needs its own scaled pass.
  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.font = '9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.shadowColor = 'rgba(0, 0, 0, .85)';
  ctx.shadowBlur = 3;
  ctx.fillStyle = 'rgba(255, 255, 255, .62)';
  ctx.strokeStyle = 'rgba(255, 255, 255, .16)';
  ctx.beginPath();
  for (const f of [2000, 4000, 6000, 8000]) {
    const y = Math.round((1 - f / FMAX) * ch) + 0.5;
    ctx.moveTo(0, y); ctx.lineTo(cw, y);
  }
  ctx.stroke();
  for (const f of [2000, 4000, 6000, 8000]) {
    const y = (1 - f / FMAX) * ch;
    ctx.fillText(f / 1000 + 'k', 5, y < 14 ? y + 11 : y - 3);
  }
  ctx.restore();
}

function tileCanvas(kind) {
  const el = tileStack && tileStack.querySelector(`.tile[data-kind="${kind}"]`);
  return el ? el.querySelector('canvas') : null;
}

function paintTile(kind) {
  const canvas = tileCanvas(kind);
  const x = waveforms[kind];
  if (!canvas || !x || x.length < NFFT) return;
  if (!specs[kind] || specs[kind].len !== x.length) {
    const frames = frameCount(x.length);
    specs[kind] = { data: spectrogram(x, frames), frames, len: x.length };
  }
  drawTile(canvas, specs[kind].data, specs[kind].frames);
}

// One tile per frame. The four spectrograms are ~50 ms of FFT work in total on
// a desktop, but a phone is several times slower and this is not why anyone
// opened the page, so it never lands in a single blocking pass.
function paintTiles(kinds = TILE_KINDS) {
  const queue = kinds.filter((k) => waveforms[k]);
  const step = () => {
    const kind = queue.shift();
    if (kind === undefined) return;
    paintTile(kind);
    if (queue.length) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function paintScaleBar() {
  const bar = document.querySelector('.scale-bar');
  if (!bar) return;
  const stops = CMAP.map((c, i) => `rgb(${c.join(',')}) ` +
    `${(i / (CMAP.length - 1) * 100).toFixed(0)}%`);
  bar.style.background = `linear-gradient(90deg, ${stops.join(', ')})`;
}

// ---------------------------------------------------------------- playhead
// All four players are time-aligned, so a single line serves the whole figure.
// currentTime is the clock: no drift, and it stays right through a seek.
let playRaf = 0;

function syncPlayhead() {
  if (!playheadEl) return;
  const el = audios().find((a) => !a.paused && !a.ended);
  if (!el) { playheadEl.style.opacity = '0'; return; }
  const w = playheadEl.parentElement.clientWidth;
  const t = Math.min(el.currentTime / (el.duration || 10), 1);
  playheadEl.style.opacity = '1';
  playheadEl.style.transform = `translateX(${(t * w).toFixed(1)}px)`;
}

function tickPlayhead() {
  syncPlayhead();
  playRaf = audios().some((a) => !a.paused && !a.ended)
    ? requestAnimationFrame(tickPlayhead) : 0;
}

function startPlayhead() {
  if (REDUCED) { syncPlayhead(); return; }
  if (!playRaf) playRaf = requestAnimationFrame(tickPlayhead);
}

const audios = () => Object.values(players).map((p) => p.audio);

// One playhead means one clock: starting any player stops the other three, so
// "which audio is the line following" is never ambiguous.
function playExclusive(el) {
  for (const o of audios()) if (o !== el && !o.paused) o.pause();
  el.play();
  startPlayhead();
}

function wirePlayers() {
  for (const a of audios()) {
    a.addEventListener('play', () => {
      for (const o of audios()) if (o !== a && !o.paused) o.pause();
      startPlayhead();
    });
    for (const ev of ['pause', 'ended', 'seeked', 'loadedmetadata']) {
      a.addEventListener(ev, syncPlayhead);
    }
    // With reduced motion there is no rAF loop; scrubbing still has to move
    // the line, and timeupdate fires often enough for that.
    if (REDUCED) a.addEventListener('timeupdate', syncPlayhead);
  }
}

// --------------------------------------------------------------------- UI
const $ = (id) => document.getElementById(id);
const fmt = (v, unit = '', digits = 2) => (Number.isFinite(v) ? v.toFixed(digits) + unit : '—');

const players = {};
document.querySelectorAll('.player').forEach((el) => {
  players[el.dataset.kind] = {
    el, canvas: el.querySelector('canvas'), audio: el.querySelector('audio'),
  };
});

let demoRef = null;         // far-end reference for metric computation
let browserOut = null;      // the waveform produced in this page
let ownOut = null;

function setStatus(text, cls = '') {
  const el = $('run-status');
  el.textContent = text;
  el.className = 'status' + (cls ? ' ' + cls : '');
}

async function bootListeners() {
  for (const [kind, p] of Object.entries(players)) {
    try {
      const wav = await readWav(p.audio.getAttribute('src'));
      waveforms[kind] = wav.data[0];
      drawWave(p.canvas, wav.data[0], COLORS[kind]);
      if (kind === 'ref') demoRef = wav.data[0];
    } catch (e) {
      console.warn('waveform failed for', kind, e);
    }
  }
  paintScaleBar();
  wirePlayers();
  window.addEventListener('resize', onResize);

  // Section 01 sits below the hero, so on a phone the figure can be a screen
  // away: nothing is computed until it is actually near the viewport.
  if (!tileStack) return;
  if (!('IntersectionObserver' in window)) { paintTiles(); return; }
  const io = new IntersectionObserver((entries, obs) => {
    if (entries.some((e) => e.isIntersecting)) { obs.disconnect(); paintTiles(); }
  }, { rootMargin: '240px' });
  io.observe(tileStack);
}

$('run-btn').addEventListener('click', async () => {
  const btn = $('run-btn');
  btn.disabled = true;
  setStatus('正在下载模型…', 'busy');
  try {
    const timer = setTimeout(() => setStatus('模型较大（6.7 MB），首次加载需要一点时间…', 'busy'), 2500);
    await getSession();
    clearTimeout(timer);

    setStatus('正在推理…', 'busy');
    const mic = await readWav('audio/nearend_mic.wav');
    const ref = await readWav('audio/farend_speech.wav');
    const t0 = performance.now();
    const res = await process(mic.data[0], ref.data[0],
                              (p) => setStatus(`正在推理… ${Math.round(p * 100)}%`, 'busy'));
    const wall = performance.now() - t0;

    browserOut = res.out;
    waveforms.out = res.out;
    delete specs.out;               // the tile has to show what the player plays
    paintTile('out');
    const preset = await readWav('audio/aec_out.wav');

    // swap the browser result into the third player
    const blob = encodeWav(res.out);
    const p = players.out;
    const url = URL.createObjectURL(blob);
    p.audio.src = url;
    drawWave(p.canvas, res.out, COLORS.out);
    p.el.classList.add('changed');

    $('m-load').textContent = `${Math.round(res.loadMs)} ms`;
    $('m-time').textContent = `${(wall / 1000).toFixed(2)} s（${res.seq} 段）`;
    $('m-rtf').textContent = (wall / 1000 / (res.out.length / SR)).toFixed(3);
    $('m-tau').textContent = `${Math.round(res.tau)} 样点（${(res.tau / SR * 1000).toFixed(1)} ms）`;
    $('m-erle').textContent = fmt(erle(mic.data[0], res.out, ref.data[0]), ' dB');
    $('m-diff').textContent = `${(maxDiff(res.out, preset.data[0]) * 32768).toFixed(0)} / 32768`;

    $('run-result').hidden = false;
    $('dl-out').href = url;
    setStatus('完成 —— 第三段音频已替换为浏览器算出的结果', 'ok');
  } catch (e) {
    console.error(e);
    setStatus('失败：' + (e && e.message ? e.message : e), 'err');
  } finally {
    btn.disabled = false;
  }
});

$('play-out').addEventListener('click', () => playExclusive(players.out.audio));

// ------------------------------------------------------------ own audio
const own = { mic: null, ref: null };

function wireUpload(inputId, nameId, key, label) {
  $(inputId).addEventListener('change', async (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    try {
      const wav = parseWav(await file.arrayBuffer());
      const mono = resample(wav.data[0], wav.rate);
      own[key] = mono;
      const secs = (mono.length / SR).toFixed(1);
      $(nameId).textContent = `${file.name} · ${wav.channels} ch · ${wav.rate} Hz → ${secs} s`;
    } catch (e) {
      own[key] = null;
      $(nameId).textContent = '读取失败：' + e.message;
    }
    $('run-own').disabled = !(own.mic && own.ref);
  });
}
wireUpload('f-mic', 'n-mic', 'mic');
wireUpload('f-ref', 'n-ref', 'ref');

$('run-own').addEventListener('click', async () => {
  const btn = $('run-own');
  btn.disabled = true;
  setStatus('正在推理你的音频…', 'busy');
  try {
    const res = await process(own.mic, own.ref,
                              (p) => setStatus(`正在推理你的音频… ${Math.round(p * 100)}%`, 'busy'));
    ownOut = res.out;
    const url = URL.createObjectURL(encodeWav(res.out));
    $('o-dur').textContent = `${(res.out.length / SR).toFixed(1)} s（${res.seq} 段）`;
    $('o-tau').textContent = `${Math.round(res.tau)} 样点（${(res.tau / SR * 1000).toFixed(1)} ms）`;
    $('o-erle').textContent = fmt(erle(own.mic, res.out, own.ref), ' dB');
    $('o-level').textContent = `${fmt(db(own.mic), ' dBFS')} → ${fmt(db(res.out), ' dBFS')}`;
    $('dl-own').href = url;
    $('own-result').hidden = false;
    $('play-own').onclick = () => new Audio(url).play();
    setStatus('完成', 'ok');
  } catch (e) {
    console.error(e);
    setStatus('失败：' + (e && e.message ? e.message : e), 'err');
  } finally {
    btn.disabled = false;
  }
});

// ------------------------------------------------------------------ start
const waveforms = {};
const specs = {};                     // kind -> dB matrix, kept across resizes

const tileStack = document.querySelector('.tile-stack');
const playheadEl = document.querySelector('.playhead');
const TILE_KINDS = tileStack
  ? [...tileStack.querySelectorAll('.tile')].map((el) => el.dataset.kind)
  : [];

const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;

async function redraw() {
  for (const [kind, samples] of Object.entries(waveforms)) {
    const p = players[kind];
    if (p) drawWave(p.canvas, samples, COLORS[kind]);
  }
}

// Resizing re-renders from the cached spectra, so a drag does not recompute
// four FFT banks per frame -- and it is debounced, because a phone fires
// resize continuously while the address bar collapses.
let resizeTimer = 0;
function onResize() {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    redraw();
    paintTiles();
    syncPlayhead();
  }, 120);
}

const COLORS = { mic: '#6ea8fe', ref: '#d2a8ff', out: '#56d364', jaec: '#e3b341' };

bootListeners();
