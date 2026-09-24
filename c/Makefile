# The engine is plain C99 and depends on nothing but libc and libm, so there is
# no build system to install and nothing to configure -- `make` is enough.
# (cpp/ needs CMake because ONNX Runtime has to be located; this does not.)
#
#   make            aec_infer
#   make test       build and run the FFT self-test
#   make libaec.a   the engine as a static library, for embedding
#   make clean

CC      ?= cc
AR      ?= ar
CFLAGS  ?= -O2
CFLAGS  += -std=c99 -Wall -Wextra
LDLIBS  += -lm

ENGINE_OBJ = engine.o fft.o wav_io.o
HEADERS    = engine.h fft.h wav_io.h

all: aec_infer

aec_infer: aec_infer.o $(ENGINE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_fft: test_fft.o fft.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

libaec.a: $(ENGINE_OBJ)
	$(AR) rcs $@ $^

# aec_infer prints wall-clock times, which needs clock_gettime; -std=c99 hides
# it behind _POSIX_C_SOURCE. The code falls back to clock() without it, so this
# is only a matter of which timer the report uses.
aec_infer.o: CFLAGS += -D_POSIX_C_SOURCE=199309L

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

# Checks the FFT against a naive DFT, then against itself. Needs no data.
test: test_fft
	./test_fft

clean:
	rm -f $(ENGINE_OBJ) aec_infer.o test_fft.o aec_infer test_fft libaec.a

.PHONY: all test clean
