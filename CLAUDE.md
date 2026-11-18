# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

zfp is a compressed-format library for multidimensional floating-point and
integer arrays, giving high-throughput random-access read/write via
compressed-array classes, plus serial/OpenMP/CUDA whole-array compression.
Core is C (C89/C99/C11-compatible); compressed-array classes are C++
header-only templates (C++98/C++11/C++14); Python (Cython) and Fortran
bindings wrap the C library.

## Build & test

Out-of-source CMake build is required (in-source builds fail fast):

```
mkdir build && cd build
cmake ..
cmake --build . --config Release
ctest
```

Key CMake options (pass as `-DOPTION=...`):
- `BUILD_TESTING_FULL=ON` — full test suite (clones GoogleTest + cmocka via
  `ExternalProject`, needs network access); plain `BUILD_TESTING` (default ON)
  only builds `testzfp`/`testviews` smoke tests.
- `BUILD_CFP=ON` — C wrapper (`cfp`) around the C++ array classes.
- `BUILD_ZFORP=ON` — Fortran bindings.
- `BUILD_ZFPY=ON` — Python bindings (`zfpy`), requires Cython/numpy.
- `BUILD_UTILITIES=ON` (default ON) — `zfp` CLI in `build/bin`.
- `BUILD_EXAMPLES=ON` — example programs in `examples/`.
- `BUILD_ALL=ON` — turns on all of the above at once.
- `ZFP_WITH_OPENMP` — auto-detected; set explicitly to force on/off.
- `ZFP_WITH_CUDA=ON` — requires CUDA toolkit; builds `src/cuda_zfp` backend.
- `ZFP_BIT_STREAM_WORD_SIZE` — 8/16/32/64 (bitstream word granularity).

Run a single test binary directly rather than through ctest when iterating,
e.g. `./build/bin/testzfp small 2d fp64`, or filter ctest by name:
`ctest -R testzfp` / `ctest --test-dir build -R <pattern>`.

A plain GNU-make build also exists (`make`, `make test`), driven by the
top-level `Config` file (uncomment/edit `BUILD_*` and `CFLAGS` there) — it is
less flexible than CMake and has no CUDA support.

Python bindings have their own build under `python/` (scikit-build-core);
see `python/CMakeLists.txt` and `pyproject.toml`.

## Architecture

### Generic C core via macro-driven file inclusion

The compression/decompression core is written once per *operation* in
`src/template/*.c` and instantiated per scalar type (`f`=float, `d`=double,
`i`=int32, `l`=int64) and per dimensionality (1–4) by small per-type driver
files that just `#include` the template bodies in sequence, e.g.
`src/encode1f.c`:

```c
#include "traitsf.h"              /* defines Scalar/Int/UInt/EBITS/... for float */
#include "template/template.h"    /* _t1/_t2 name-mangling macros */
#include "template/codec.h"
#include "zfp/bitstream.inl"
#include "template/codec.c"
#include "template/codecf.c"      /* float-specific codec bits */
#include "template/codec1.c"      /* 1D-specific codec bits */
#include "template/encode.c"
#include "template/encodef.c"
#include "template/encode1.c"
```

`src/traits{f,d,i,l}.h` `#define` the scalar/integer types and
exponent/precision bit counts for each data type; `src/block{1,2,3,4}.h`
define per-dimension block layout constants. `src/template/template.h`
provides the `_cat2`/`_cat3`/`_t1`/`_t2` token-pasting macros used to generate
per-type/per-dim symbol names (e.g. `encode_block_float_1`). When changing
core codec logic, edit the shared `template/*.c` file, not the generated
`encode<dim><type>.c`/`decode<dim><type>.c` files — those are just inclusion
shims. `src/template/rev*.c` implement the newer "reversed"/row-major
encode/decode path used for CUDA-compatible access.

`src/zfp.c` is the public C API entry point (`zfp_compress`,
`zfp_decompress`, stream/field setup, mode selection: fixed-rate,
fixed-precision, fixed-accuracy, reversible). `src/bitstream.c` implements
the low-level bitstream reader/writer.

### CUDA backend

`src/cuda_zfp/` is a self-contained, independently-templated reimplementation
of the codec for the GPU — it does **not** reuse `src/template/*.c`; it
mirrors the same algorithm in C++/CUDA template form instead:

- `cuZFP.h` declares the only two entry points exposed to the C core:
  `cuda_compress(zfp_stream*, const zfp_field*)` and
  `cuda_decompress(zfp_stream*, zfp_field*)` (`extern "C"`, so the rest of
  the codebase can call into it without knowing it's C++/CUDA underneath).
- `cuZFP.cu` is the only `.cu` translation unit (built via
  `cuda_wrap_srcs` in `src/CMakeLists.txt`, not as a normal source file) and
  `#include`s the `.cuh` headers directly rather than linking separate
  objects — the whole backend compiles as one NVCC unit. It also pulls in
  `zfp/bitstream.inl` directly (guarding against the `inline_` macro
  colliding with the C core's) so its bitstream reads/writes stay
  bit-compatible with the serial/OpenMP encoders.
- Dimensionality is handled by **separate** files per dimension —
  `encode1.cuh`/`decode1.cuh`, `encode2.cuh`/`decode2.cuh`,
  `encode3.cuh`/`decode3.cuh` — templated on scalar type, dispatched at
  runtime in `cuZFP.cu`'s `internal::encode<T>`/`internal::decode<T>` by
  inspecting how many of `field->nx/ny/nz` are nonzero. **There is no
  `encode4.cuh`/`decode4.cuh`: CUDA compression does not support 4D arrays**,
  and `cudacompress.c`/`cudadecompress.c` (below) only instantiate wrappers
  for dims 1–3.
- `type_info.cuh` is the CUDA-side analogue of `src/traits{f,d,i,l}.h`:
  `__host__ __device__` template specializations (`get_ebias<T>`,
  `get_ebits<T>`, `get_precision<T>`, `zfp_traits<T>::{Int,UInt}`, etc.) for
  `float`/`double`/`int`/`long long int`, used by the device encode/decode
  kernels instead of the macro-based traits headers.
- `pointers.cuh` provides `is_gpu_ptr()` (via `cudaPointerGetAttributes`),
  which both entry points use to decide whether `field->data` /
  `stream->stream->begin` already live on the device. If not, `cuZFP.cu`'s
  `setup_device_field_*`/`setup_device_stream_*` helpers `cudaMalloc` +
  `cudaMemcpy` a staging buffer and `cleanup_device_ptr` copies the result
  back and frees it afterwards — so callers can pass either host or device
  pointers transparently, but non-contiguous *host* arrays are rejected
  (`is_contigous{1,2,3}d` checks; `cuda_compress` returns 0 if the host field
  isn't contiguous).
- `shared.h` / `ErrorCheck.h` hold shared device-side constants and a
  `cudaGetLastError`-checking helper (`ErrorCheck::chk`) called after each
  kernel launch.
- CUDA compression is currently **fixed-rate mode only**: the
  `compress_cuda_*`/`decompress_cuda_*` wrappers each check
  `zfp_stream_compression_mode(stream) == zfp_mode_fixed_rate` before calling
  into `cuZFP.cu`, and silently no-op otherwise — a request to CUDA-compress
  in fixed-precision/fixed-accuracy/reversible mode produces no output
  without an explicit error.

Wiring into the generic C core happens through the same macro-generated
function-table mechanism as OpenMP: `src/template/cudacompress.c` /
`cudadecompress.c` are included from `src/zfp.c` once per scalar type
(guarded by `#ifdef ZFP_WITH_CUDA`) and generate
`compress_cuda_<type>_<dim>` / `compress_strided_cuda_<type>_<dim>`
functions (and `decompress_*` equivalents) that just forward to
`cuda_compress`/`cuda_decompress`. `zfp.c` then slots these into its
`[execution][strided][dimensionality][scalar type]` dispatch table (see
`zfp_compress`/`zfp_decompress` around the `zfp_exec_cuda` case), selected at
runtime via `zfp_stream_set_execution(stream, zfp_exec_cuda)`.

The backend is compiled into the main `zfp` library target only when
`ZFP_WITH_CUDA=ON` (see `src/CMakeLists.txt`: `cuda_wrap_srcs(zfp OBJ
zfp_cuda_backend_obj cuda_zfp/cuZFP.cu ...)`, linked against
`CUDA_CUDART_LIBRARY` and `stdc++`) — there is no separate CUDA library
target to build or link against independently.

### C++ compressed-array classes

`include/zfp/array.hpp` and `array{1,2,3,4}.hpp` (plus `constarray*.hpp` for
read-only/view variants) are the public, header-only C++ API — these are
templates over scalar type and dimension, not generated per-type. They wrap
the C codec and add a block cache (`include/zfp/internal/array/cache*.hpp`)
providing high-throughput random element access without fully decompressing
the array. `include/zfp/factory.hpp` and `index.hpp` support dynamic
dispatch and compressed-array indexing (variable-rate mode).
`include/zfp/internal/` holds implementation details not part of the public
API surface (`internal/zfp`, `internal/array`, `internal/cfp`,
`internal/codec`).

### CFP and Fortran bindings

`cfp/` is a C-callable wrapper around the C++ array classes (one `.cpp` per
type/dimension, e.g. `cfparray2d.cpp`), built when `BUILD_CFP=ON`; it exists
so non-C++ callers (including Fortran) can use compressed arrays.
`fortran/zfp.f90` is the Fortran 90 interface, built when `BUILD_ZFORP=ON`.

### Tests

- `tests/testzfp.cpp` / `tests/testviews.cpp` — smoke tests always built
  with `BUILD_TESTING` (no external deps); registered as ctest `testzfp`,
  `testviews`.
- `BUILD_TESTING_FULL=ON` additionally builds: `tests/src/{encode,decode,
  inline,misc,endtoend,execPolicy}` (cmocka-based C unit tests against the
  codec internals, one subdir per concern) and `tests/array` (GoogleTest-based
  C++ tests of the array classes, under `array/`, `constArray/`, `decode/`,
  `encode/`, `zfp/`). Both cmocka and GoogleTest are fetched automatically via
  `ExternalProject`/CMake FetchContent-style download at configure time —
  expect that step to need network access the first time.
- `tests/constants/` holds precomputed checksums and golden random-data
  headers per type/dimension (`tests/constants/checksums`,
  `tests/utils/genChecksums.sh` regenerates them) — regression tests compare
  compressed output against these; a core codec change that's intentional
  will usually require regenerating checksums.
- `tests/utils/` has shared test helpers (hashing, RNG, strided-array
  helpers, timers) linked into the cmocka tests.
- Small vs. large array regression tests are gated by
  `ZFP_BUILD_TESTING_SMALL` / `ZFP_BUILD_TESTING_LARGE` (`large` is off by
  default — it's slow).

### Compile-time configuration surface

Many behavioral knobs are CMake options turned into preprocessor defines
rather than runtime flags (see `CMakeLists.txt` "Top level options"):
`ZFP_WITH_TIGHT_ERROR`, `ZFP_ROUNDING_MODE`, `ZFP_WITH_DAZ`,
`ZFP_WITH_ALIGNED_ALLOC`, `ZFP_WITH_CACHE_TWOWAY`,
`ZFP_WITH_CACHE_FAST_HASH`, `ZFP_WITH_CACHE_PROFILE`,
`ZFP_WITH_BIT_STREAM_STRIDED`. When debugging numerical or cache behavior,
check whether it's gated behind one of these before assuming it's a runtime
code path.

## Versioning / branching

Development happens on `develop` (Gitflow); PRs target `develop`. `master`
tracks released versions only. Version numbers are parsed out of
`include/zfp/version.h` by the top-level `CMakeLists.txt` — bump there, not
in CMakeLists.
