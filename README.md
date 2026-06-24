# Building cgeist (HLS-pragma fork)

This is a fork of Polygeist's `cgeist` (Clang → MLIR frontend) modified to **ingest
HLS pragmas** and attach them as MLIR attributes for downstream BRAM/DSP/LUT
estimation. Only `cgeist` matters here — the rest of Polygeist (polygeist-opt, GPU
backends, polymer) is not needed for this flow and is ignored below.

Supported pragmas: `array_partition`, `array_reshape`, `bind_storage` (storage-bound,
resolved against the alloca), `bind_op` (computation-bound, resolved via
`polygeist.ssa_names`), and `pipeline` / `unroll` / `allocation` (loop/function-bound,
attached to the resolved scope op).

## Requirements
- C/C++ toolchain, cmake, ninja (`sudo apt install ninja-build` on a fresh box)
- Linux / WSL2 (developed on WSL2 Ubuntu 22.04)


## 1. Build the pinned LLVM / MLIR / Clang

cgeist links against these, so build them first from Polygeist's nested submodule:

```sh
mkdir Polygeist/llvm-project/build
cd Polygeist/llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_USE_SPLIT_DWARF=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
ninja
```

## 2. Build cgeist

Configure Polygeist against that LLVM build, then build just the `cgeist` target:

```sh
mkdir Polygeist/build
cd Polygeist/build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
  -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
  -DCMAKE_C_COMPILER=$PWD/../llvm-project/build/bin/clang \
  -DCMAKE_CXX_COMPILER=$PWD/../llvm-project/build/bin/clang++ \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
ninja cgeist
```

`ninja cgeist` builds the single tool — no need to build all of Polygeist.

Binary lands at `Polygeist/build/bin/cgeist`.


## Driving HLS annotation: `--hls-annotate`

The HLS pragma handling is gated by the **`--hls-annotate`** cgeist flag. Pragmas are
always parsed, but resolution + attribute attachment only runs when this flag is set —
and it runs in `driver.cc` *after* the pass pipeline has lifted CFG → SCF → affine, so
the pragmas attach to real `affine.for` / `memref.alloca` ops rather than raw `cf.br`.

Canonical invocation:

```sh
cgeist input.cpp --function="*" -S --hls-annotate --raise-scf-to-affine --memref-fullrank
```

- **With** `--hls-annotate`: allocas/loops/funcs carry `hls.*` attributes, e.g.
  `hls.array_partition = [{kind = "complete", variable = "arr"}]`
- **Without** it: cgeist behaves like stock, pragmas are ignored

Note `--function="*"` for C++ input: a specific name like `--function=top` won't match
the mangled symbol (`_Z3topv`), giving an empty `module {}`.

## Tests / examples

Lit-based examples live in **`/tools/cgeist/Test/HLS`**  these are the reference for
the expected `hls.*` attribute output per pragma. Run them with the dedicated target:

```sh
ninja check-cgeist
```

Each test is a self-contained `RUN`/`CHECK` pair driven by `--hls-annotate`, e.g.:

```
//RUN: cgeist %s --function="*" -S --hls-annotate --raise-scf-to-affine --memref-fullrank| FileCheck %s

void basic_array_partition() {
    int arr[8];
#pragma HLS array_partition variable=arr complete
    // Simple computation to test parallel access
    for(int i = 0; i < 8; i++) {
        arr[i] += 1;
    }
}

// CHECK: %alloca = memref.alloca() {hls.array_partition = [{kind = "complete", variable = "arr"}],
```

For quick manual checks outside lit, the `basic_test/` kernels + `convert.sh` wrapper
run cgeist and dump the annotated MLIR directly.

(Adding a new `test/cgeist/` case requires `cgeist` in the `DEPENDS` list of
`test/CMakeLists.txt` and registered in `test/lit.cfg.py`'s tools so it's on PATH.)
