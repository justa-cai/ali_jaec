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
  window.addEventListener('resize', redraw);
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
    const preset = await readWav('audio/aec_out.wav');

    // swap the browser result into the third player
    const blob = encodeWav(res.out);
    const p = players.out;
    const url = URL.createObjectURL(blob);
    p.audio.src = url;
    drawWave(p.canvas, res.out, '#56d364');
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

$('play-out').addEventListener('click', () => players.out.audio.play());

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

async function redraw() {
  for (const [kind, samples] of Object.entries(waveforms)) {
    drawWave(players[kind].canvas, samples, COLORS[kind]);
  }
}
const COLORS = { mic: '#6ea8fe', ref: '#d2a8ff', out: '#56d364' };

bootListeners();
