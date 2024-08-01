use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::ops::Mul;
use std::{collections::HashMap, io::Write};

fn main() {
    let mut funcs: HashMap<&'static str, fn(u128) -> u128> = HashMap::new();
    funcs.insert("test_singlthreaded_normal", fib_st_normal);
    funcs.insert("test_singlethreaded_memo", fib_st_memo);
    funcs.insert("test_singlethreaded_linear", fib_st_linear);
    funcs.insert("test_singlethreaded_matrix", fib_st_matrix);
    funcs.insert("test_singlethreaded_matrix_expo", fib_st_matrix_expo);

    rayon::ThreadPoolBuilder::new()
        .num_threads(funcs.len())
        .build_global()
        .expect("Failed to limit the thread counts");

    let a = funcs
        .par_iter()
        .map(|(k, v)| (k.to_string(), test(v, k)))
        .collect::<Vec<(String, u128)>>();

    for x in a {
        println!(
            "Result for {} took number {} for fibbonaci operation to take more than a second",
            x.0, x.1
        )
    }
}

#[must_use]
fn test<T>(call: &dyn Fn(u128) -> T, test_name: &str) -> u128 {
    let mut x = 0;
    println!("\nTesting {}", test_name);
    loop {
        let now = std::time::Instant::now();
        call(x);
        let ex_time = std::time::Instant::now();
        let a = (ex_time - now).as_secs_f32();
        if a > 1.0 {
            // changing this to 0.5 so i dont have to wait a day
            break;
        }
        if (x % 100000) == 0 {
            print!("\rX ({test_name}) is now: {x}                 ");
            std::io::stdout().flush().expect("Failed to flush");
        }
        x += 1;
    }
    println!(
        "\n({test_name}) The number that took more than one second is: {}",
        x
    );
    x
}

fn fib_st_normal(n: u128) -> u128 {
    match n {
        0 | 1 => n,
        _ => fib_st_normal(n - 1) + fib_st_normal(n - 2),
    }
}

lazy_static::lazy_static! {
    static ref CACHE: std::sync::Mutex<std::collections::HashMap<u128, u128>> = {
        std::sync::Mutex::new(std::collections::HashMap::new())
    };
}

fn fib_st_memo(n: u128) -> u128 {
    match n {
        0 | 1 => n,
        _ => *CACHE
            .lock()
            .unwrap()
            .entry(n)
            .or_insert(fib_st_normal(n - 1) + fib_st_normal(n - 2)),
    }
}

fn fib_st_linear(mut n: u128) -> u128 {
    let mut a = 0;
    let mut b = 1;
    let mut next;

    while n > 0 {
        n -= 1;
        next = a + b;
        a = b;
        b = next;
    }
    a
}

#[derive(Clone, Copy)]
struct Matrix2x2(u128, u128, u128, u128);

impl Mul for Matrix2x2 { // mulassign bad
    type Output = Matrix2x2;
    fn mul(self, rhs: Self) -> Self::Output {
        let a = self.0 * rhs.0 + self.1 * rhs.2;
        let b = self.0 * rhs.1 + self.1 * rhs.3;
        let c = self.2 * rhs.0 + self.3 * rhs.2;
        let d = self.2 * rhs.1 + self.3 * rhs.3;
        Matrix2x2(a,b,c,d)
    }
}

fn fib_st_matrix(mut n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    let step = Matrix2x2(0,1,1,1);
    let mut fib = Matrix2x2(0,1,1,1);
    while n > 0 {
        n -= 1;
        fib = fib * step;
    }
    fib.0
}

fn fib_st_matrix_expo(mut n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    let mut fib = Matrix2x2(1, 0, 0, 1); // cpp implementation doesn't produce it correctly so here's my changes i guess
    let mut step = Matrix2x2(1, 1, 1, 0);
    n -= 1;
    while n > 0 {
        if (n & 1) != 0 {
            fib = fib * step;
        }
        step = step * step;
        n >>= 1;
    }
    fib.0
}
#[cfg(test)]
mod tests {
    use crate::{fib_st_linear, fib_st_matrix, fib_st_matrix_expo, fib_st_memo, fib_st_normal};

    const FIB: u128 = 40;
    const TARGET_VALUE: u128 = 102334155;

    #[test]
    fn test_normal() {
        assert_eq!(fib_st_normal(FIB), TARGET_VALUE)
    }

    #[test]
    fn test_memo() {
        assert_eq!(fib_st_memo(FIB), TARGET_VALUE)
    }

    #[test]
    fn test_linear() {
        assert_eq!(fib_st_linear(FIB), TARGET_VALUE)
    }

    #[test]
    fn test_matrix() {
        assert_eq!(fib_st_matrix(FIB), TARGET_VALUE)
    }

    #[test]
    fn test_matrix_expo() {
        assert_eq!(fib_st_matrix_expo(FIB), TARGET_VALUE)
    }
}