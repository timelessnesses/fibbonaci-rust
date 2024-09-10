/**
  * No memoization sadly
 */

int fib_gpu_normal_real(int n) {
    if (n == 0 || n == 1) {
        return n;
    }
    return fib_gpu_normal_real(n - 1) + fib_gpu_normal_real(n - 2);
}

// crashes my gpu
kernel void fib_gpu_normal(global ulong* results, ulong n) {
    int id = get_global_id(0);
    if (id < n) {
        results[id] = fib_gpu_normal_real(n);
    }
}

// works!! hooray!!!
kernel void fib_gpu_linear(global ulong* results, ulong n) {
    int id = get_global_id(0);

    if (id < n) {
        ulong a = 0;
        ulong b = 1;
        ulong next;

        ulong m = n;
        while (m > 0) {
            m -= 1;
            next = a + b;
            a = b;
            b = next;
        }
        results[id] = a;
    }
}

typedef struct {
    ulong elements[4];
} Matrix2x2;

Matrix2x2 matrix_mul(Matrix2x2 m1, Matrix2x2 m2) {
    Matrix2x2 res;
    res.elements[0] = m1.elements[0] * m2.elements[0] + m1.elements[1] * m2.elements[2];
    res.elements[1] = m1.elements[0] * m2.elements[1] + m1.elements[1] * m2.elements[3];
    res.elements[2] = m1.elements[2] * m2.elements[0] + m1.elements[3] * m2.elements[2];
    res.elements[3] = m1.elements[2] * m2.elements[1] + m1.elements[3] * m2.elements[3];
    return res;
}

// data buffer exceed length?
kernel void fib_gpu_matrix(global ulong* results, ulong n) {
    int id = get_global_id(0);

    if (id < n) {
        Matrix2x2 gpuep = {{0, 1, 1, 1}};
        Matrix2x2 fib = {{0, 1, 1, 1}};

        ulong m = n;
        while (m > 0) {
            m -= 1;
            fib = matrix_mul(fib, gpuep);
        }
        results[id] = fib.elements[0];
    }
}

// returns fucking 0
kernel void fib_gpu_matrix_expo(global ulong* results, ulong n) {
    int id = get_global_id(0);

    if (id < n) {
        if (n == 0) {
            results[id] = 0;
            return;
        } else if (n == 1) {
            results[id] = 1;
            return;
        }

        Matrix2x2 fib = {{1, 0, 0, 1}};
        Matrix2x2 gpuep = {{1, 1, 1, 0}};

        ulong m = n - 1;
        while (m > 0) {
            if (m & 1) {
                fib = matrix_mul(fib, gpuep);
            }
            gpuep = matrix_mul(gpuep, gpuep);
            m >>= 1;
        }
        results[id] = fib.elements[0];
    }
}