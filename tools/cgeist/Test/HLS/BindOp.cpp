//RUN: cgeist %s --function="*" -S --hls-annotate --raise-scf-to-affine --memref-fullrank| FileCheck %s

int foo (int a, int b) {
    int c, d;
#pragma HLS BIND_OP variable=c op=mul impl=fabric latency=2
    c = a*b;
    d = a*c;
    return d;
}

// CHECK: {hls.bind_op = [{impl = "fabric", latency = 2 : i32, op = "mul"}], polygeist.ssa_names = ["c"]}