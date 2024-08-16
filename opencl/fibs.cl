/**
  * No memoization sadly
 */

int fib_st_normal_real(int n) {
    if (n == 0 || n == 1) {
        return n;
    }
    return fib_st_normal_real(n - 1) + fib_st_normal_real(n - 2);
}

kernel void fib_st_normal(global ulong* results, ulong n) {
    int id = get_global_id(0);
    if (id < n) {
        results[id] = fib_st_normal_real(id);
    }
}

kernel void fib_st_linear(global ulong* results, ulong n) {
    int id = get_global_id(0);

    if (id < n) {
        ulong a = 0;
        ulong b = 1;
        ulong next;

        ulong m = id;
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
    ulong a, b, c, d;
} Matrix2x2;

Matrix2x2 matrix_mul(Matrix2x2 m1, Matrix2x2 m2) {
    Matrix2x2 res;
    res.a = m1.a * m2.a + m1.b * m2.c;
    res.b = m1.a * m2.b + m1.b * m2.d;
    res.c = m1.c * m2.a + m1.d * m2.c;
    res.d = m1.c * m2.b + m1.d * m2.d;
    return res;
}

kernel void fib_st_matrix(global ulong* results, ulong n) {
    int id = get_global_id(0);

    if (id < n) {
        Matrix2x2 step = {0, 1, 1, 1};
        Matrix2x2 fib = {0, 1, 1, 1};

        ulong m = id;
        while (m > 0) {
            m -= 1;
            fib = matrix_mul(fib, step);
        }
        results[id] = fib.a;
    }
}

kernel void fib_st_matrix_expo(global ulong* results, ulong n) {
    int id = get_global_id(0);

    if (id < n) {
        if (id == 0) {
            results[id] = 0;
            return;
        }

        Matrix2x2 fib = {1, 0, 0, 1};
        Matrix2x2 step = {1, 1, 1, 0};

        ulong m = id - 1;
        while (m > 0) {
            if (m & 1) {
                fib = matrix_mul(fib, step);
            }
            step = matrix_mul(step, step);
            m >>= 1;
        }
        results[id] = fib.a;
    }
}