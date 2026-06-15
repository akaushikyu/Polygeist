//RUN: cgeist %s --function="*" -S --hls-annotate --raise-scf-to-affine --memref-fullrank| FileCheck %s


void my_kernel(int j) {
    j = 0;
    for(int i = 0; i < 10; ++i) {
        #pragma HLS pipeline II=1
        j += 1;
    }
}

// CHECK: hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]