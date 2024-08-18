# fibs

just very fast fibbonaci functions (TC: Time Complexity, SC: Space Complexity)

## order

1. `fib_st_matrix_expo` (TC: O(log(n)), SC: O(1)) [https://www.youtube.com/watch?v=KzT9I1d-LlQ](https://www.youtube.com/watch?v=KzT9I1d-LlQ)
2. `fib_st_matrix` (TC: O(n), SC: O(1)) [https://www.youtube.com/watch?v=KzT9I1d-LlQ](https://www.youtube.com/watch?v=KzT9I1d-LlQ)
3. `fib_st_linear` (TC: O(n), SC: O(1))
4. `fib_st_memo` (TC: O(n), SC: O(n))
5. `fib_gpu_matrix` (TC: O(n), SC: O(1))
6. `fib_gpu_matrix_expo` (TC: O(log(n)), SC: O(1))
7. `fib_gpu_linear` (TC: O(n), SC: O(1))
8. `fib_st_normal` (TC: O(2^n), SC: O(n))
9. `fib_gpu_normal` (TC: O(2^n), SC: O(n)) (crashing my gpu)

## note

`fib_gpu_matrix` and `fib_gpu_matrix_expo` seems to struggles and hiccup from time to time while `fib_gpu_linear` is actually has stable generation rate and doesn't hiccup.
