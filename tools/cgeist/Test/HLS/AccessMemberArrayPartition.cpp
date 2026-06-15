//RUN: cgeist %s --function="*" -S --hls-annotate --raise-scf-to-affine --memref-fullrank| FileCheck %s

struct C {
    int A[8];
    int B[4];
};

struct S {
    int A[8];
    int B[4];
    C c;
};

int top(int input1[8], int input2[4]) {
    S s;
#pragma HLS array_partition variable=s.A complete
#pragma HLS array_partition variable=s.c.A complete
#pragma HLS array_partition variable=s.B block factor=2
    for (int i = 0; i < 8; i++) {
        s.A[i] = input1[i];
        s.c.A[i] = 2 * input1[i];
    }
    for (int i = 0; i < 4; i++) {
        s.B[i] = input2[i];
        s.c.B[i] = 2 * input2[i];
    }

    for (int i = 0; i < 8; i++) {
        s.A[i] = s.A[i] + 1;
        s.c.A[i] = s.c.A[i] + 1;
    }
    int sum = 0;
    for (int i = 0; i < 8; i++) {
        sum += s.A[i];
        sum += s.c.A[i];
    }

    return sum;

}

// CHECK: %alloca = memref.alloca() {hls.array_partition = [{kind = "complete", variable = "s.A"}, {kind = "complete", variable = "s.c.A"}, {factor = 2 : i32, kind = "block", variable = "s.B"}], polygeist.varname = "s"}

