use dashmap::DashMap;
use deepsize::DeepSizeOf;
use flume::bounded;
use lazy_static::lazy_static;
use ocl::ProQue;
use ratatui::crossterm::terminal::disable_raw_mode;
use ratatui::crossterm::terminal::enable_raw_mode;
use ratatui::crossterm::terminal::EnterAlternateScreen;
use ratatui::crossterm::terminal::LeaveAlternateScreen;
use ratatui::crossterm::ExecutableCommand;
use ratatui::prelude::CrosstermBackend;
use ratatui::prelude::*;
use ratatui::widgets::Block;
use ratatui::widgets::Borders;
use ratatui::widgets::Paragraph;
use ratatui::Frame;
use ratatui::Terminal;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::collections::HashMap;
use std::io::stdout;
use std::ops::Mul;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

lazy_static! {
    static ref PRO_QUE: Arc<Mutex<ProQue>> = Arc::new(Mutex::new(
        ProQue::builder()
            .src(include_str!("../opencl/fibs.cl"))
            .dims(1)
            .build()
            .expect("Failed to build proque")
    ));
    static ref CACHE_H: std::sync::RwLock<std::collections::HashMap<u64, u64>> =
        std::sync::RwLock::new(std::collections::HashMap::new());
    static ref CACHE_V: std::sync::RwLock<Vec<u64>> = std::sync::RwLock::new(Vec::new());
}

const TIMEOUT: f32 = 1.0;
const HARD_LIMIT: usize = 1_000_000;

fn main() {
    let dies = Arc::new(AtomicBool::new(false));
    let cloned_death = Arc::clone(&dies);
    ctrlc::set_handler(move || {
        cloned_death.store(true, std::sync::atomic::Ordering::Relaxed);
    })
    .expect("Failed to set CTRL C handler");
    let mut funcs: HashMap<String, Arc<dyn Fn(u64) -> u64 + Send + Sync>> = HashMap::new();
    funcs.insert(
        "test_singlethreaded_normal".to_string(),
        Arc::new(fib_st_normal),
    );
    funcs.insert(
        "test_singlethreaded_memo_hashmap".to_string(),
        Arc::new(fib_st_memo_hashmap),
    );
    funcs.insert(
        "test_singlethreaded_memo_vec".to_string(),
        Arc::new(fib_st_memo_vec),
    );
    funcs.insert(
        "test_singlethreaded_linear".to_string(),
        Arc::new(fib_st_linear),
    );
    funcs.insert(
        "test_singlethreaded_matrix".to_string(),
        Arc::new(fib_st_matrix),
    );
    funcs.insert(
        "test_singlethreaded_matrix_expo".to_string(),
        Arc::new(fib_st_matrix_expo),
    );
    funcs.insert(
        "test_singlethreaded_successor".to_string(),
        Arc::new(fib_st_successor),
    );
    // please don't uncomment gpu_normal
    // funcs.insert("test_gpu_normal".to_string(), Arc::new(fib_normal_gpu_wrapper()));
    funcs.insert(
        "test_gpu_linear".to_string(),
        Arc::new(fib_linear_gpu_wrapper()),
    );
    funcs.insert(
        "test_gpu_matrix".to_string(),
        Arc::new(fib_matrix_gpu_wrapper()),
    );
    funcs.insert(
        "test_gpu_matrix_expo".to_string(),
        Arc::new(fib_matrix_expo_gpu_wrapper()),
    );
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(funcs.len())
    //     .build_global()
    //     .expect("Failed to limit the thread counts");

    let mut h = HashMap::new();
    for (name, func) in funcs.iter() {
        let (sender, receiver) = bounded::<(u64, f32)>(1);
        test(func.clone(), sender);
        h.insert(name.clone(), receiver);
    }
    run_tui(h, dies).unwrap();
}

fn run_tui(
    tests: HashMap<String, flume::Receiver<(u64, f32)>>,
    dies: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    let progress = Arc::new(DashMap::new());

    let cloned = Arc::clone(&progress);
    let demise = dies.clone();
    std::thread::spawn(move || loop {
        tests.par_iter().for_each(|(name, r)| {
            if let Ok(a) = r.try_recv() {
                if a.0 != 0 {
                    cloned
                        .entry(name.to_string())
                        .and_modify(|i: &mut (u64, bool, f32)| {
                            i.0 = a.0;
                            i.2 = a.1;
                        })
                        .or_insert((a.0, false, a.1));
                    return;
                }
                cloned.entry(name.to_string()).and_modify(|i| {
                    i.1 = true;
                    i.2 = a.1;
                });
            }
            if r.is_disconnected() {
                cloned.entry(name.to_string()).and_modify(|i| i.1 = true);
            }
        });
        if demise.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
    });

    enable_raw_mode().unwrap();
    stdout().execute(EnterAlternateScreen).unwrap();

    while !dies.load(std::sync::atomic::Ordering::Relaxed) {
        terminal
            .draw(|f| {
                draw_ui(f, &progress);
            })
            .unwrap();
        if dies.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    disable_raw_mode().unwrap();
    stdout().execute(LeaveAlternateScreen).unwrap();

    Ok(())
}

fn draw_ui(frame: &mut Frame, progress: &Arc<DashMap<String, (u64, bool, f32)>>) {
    match progress.len() {
        0 => {
            let paragraph = Paragraph::new("Initializing")
                .block(Block::default().title("Initializing tests..."));
            frame.render_widget(
                paragraph,
                Layout::default()
                    .constraints(vec![Constraint::Percentage(100)])
                    .split(frame.area())[0],
            )
        }
        _ => {
            let mut sorted_progress: Vec<_> = progress.iter().collect();
            sorted_progress.sort_by(|a, b| b.value().2.partial_cmp(&a.value().2).unwrap());

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![
                    Constraint::Percentage(100 / progress.len() as u16 + 1);
                    progress.len() + 1
                ])
                .split(frame.area());

            for (i, entry) in sorted_progress.iter().enumerate() {
                let test_name = entry.key();
                let prog = entry.value();
                let text = format!(
                    "Status: Solving Fibonacci {} ({})\nSpeed: {} Fibonacci numbers/second",
                    prog.0,
                    {
                        if prog.1 {
                            if prog.0 != HARD_LIMIT as u64 {
                                "Timed out"
                            } else {
                                "Done"
                            }
                        } else {
                            "Running"
                        }
                    },
                    prog.2
                );

                let paragraph = Paragraph::new(text).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(format!("#{} ", i + 1) + test_name)
                        .border_style(match prog.1 {
                            true => match prog.0 as usize {
                                HARD_LIMIT => Color::Green,
                                _ => Color::Red,
                            },
                            false => Color::Cyan,
                        }),
                );

                frame.render_widget(paragraph, chunks[i]);
            }

            let cache_stat = Paragraph::new(format!(
                "CACHE Hashmap Size is currently: {} megabytes\nCACHE Vec Size is currently: {} megabytes\n\nTimeout is: {} seconds",
                CACHE_H.deep_size_of() as f32 / 1_048_576_f32,
                CACHE_V.deep_size_of() as f32 / 1_048_576_f32,
                TIMEOUT
            ))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Memory status"),
            );
            frame.render_widget(cache_stat, *chunks.last().unwrap());
        }
    }
}

fn test(call: Arc<dyn Fn(u64) -> u64 + Send + Sync>, tx: flume::Sender<(u64, f32)>) {
    rayon::spawn(move || {
        let mut x = 1;
        let mut previous_fns = 0.0;
        loop {
            let now = std::time::Instant::now();
            call(x);
            let ex_time = std::time::Instant::now();
            let duration = (ex_time - now).as_secs_f32();
            if duration > TIMEOUT {
                break;
            }
            previous_fns = 1.0 / duration;
            tx.send((x, previous_fns)).expect("Failed to send data");
            x += 1;
            if x > HARD_LIMIT as u64 {
                break;
            }
        }
        tx.send((0, previous_fns))
            .expect("Failed to send final progress"); // mark as done
    });
}

#[allow(dead_code)]
fn fib_normal_gpu_wrapper() -> impl Fn(u64) -> u64 + 'static {
    let pro_que = PRO_QUE.clone();
    move |i: u64| {
        let pro_que = pro_que.lock().unwrap();
        let results = pro_que
            .create_buffer::<u64>()
            .expect("Failed to build result buffer");
        let kernel = pro_que
            .kernel_builder("fib_gpu_normal")
            .arg(&results)
            .arg(i)
            .build()
            .expect("Failed to build kernel");
        unsafe {
            kernel.enq().expect("Failed to execute");
        };
        let mut b = vec![0u64; 1];
        results.read(&mut b).enq().expect("Failed to read");
        b[0]
    }
}

fn fib_linear_gpu_wrapper() -> impl Fn(u64) -> u64 + 'static {
    let pro_que = PRO_QUE.clone();
    move |i: u64| {
        let pro_que = pro_que.lock().unwrap();
        let results = pro_que
            .create_buffer::<u64>()
            .expect("Failed to build result buffer");
        let kernel = pro_que
            .kernel_builder("fib_gpu_linear")
            .arg(&results)
            .arg(i)
            .build()
            .expect("Failed to build kernel");
        unsafe {
            kernel.enq().expect("Failed to execute");
        };
        let mut b = vec![0u64; 1];
        results.read(&mut b).enq().expect("Failed to read");
        b[0]
    }
}

fn fib_matrix_gpu_wrapper() -> impl Fn(u64) -> u64 + 'static {
    let pro_que = PRO_QUE.clone();
    move |i: u64| {
        let pro_que = pro_que.lock().unwrap();
        let mut results = ocl::Buffer::builder()
            .len(1)
            .queue(pro_que.queue().clone())
            .build()
            .unwrap();
        let kernel = pro_que
            .kernel_builder("fib_gpu_matrix")
            .arg(&mut results)
            .arg(i)
            .global_work_size(1)
            .build()
            .expect("Failed to build kernel");
        unsafe {
            kernel.enq().expect("Failed to execute");
        };
        let mut b = vec![0u64; 1];
        results.read(&mut b).enq().expect("Failed to read");
        b[0]
    }
}

fn fib_matrix_expo_gpu_wrapper() -> impl Fn(u64) -> u64 + 'static {
    let pro_que = PRO_QUE.clone();
    move |i: u64| {
        let pro_que = pro_que.lock().unwrap();
        let mut results = ocl::Buffer::builder()
            .len(1)
            .queue(pro_que.queue().clone())
            .build()
            .unwrap();
        let kernel = pro_que
            .kernel_builder("fib_gpu_matrix_expo")
            .arg(&mut results)
            .arg(i)
            .global_work_size(1)
            .build()
            .expect("Failed to build kernel");
        unsafe {
            kernel.enq().expect("Failed to execute");
        };
        let mut b = vec![0u64; 1];
        results.read(&mut b).enq().expect("Failed to read");
        b[0]
    }
}

fn fib_st_normal(n: u64) -> u64 {
    match n {
        0 | 1 => n,
        _ => fib_st_normal(n - 1) + fib_st_normal(n - 2),
    }
}

fn fib_st_memo_hashmap(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    {
        let cache = CACHE_H.read().unwrap();
        if let Some(&result) = cache.get(&n) {
            return result;
        }
    }
    let result = fib_st_memo_hashmap(n - 1) + fib_st_memo_hashmap(n - 2);
    {
        let mut cache = CACHE_H.write().unwrap();
        cache.insert(n, result);
    }
    result
}

fn fib_st_memo_vec(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    {
        let cache = CACHE_V.read().unwrap();
        if (n as usize) < cache.len() {
            return cache[n as usize - 1];
        }
    }
    let result = fib_st_memo_vec(n - 1) + fib_st_memo_vec(n - 2);
    {
        let mut cache = CACHE_V.write().unwrap();
        if n as usize >= cache.len() {
            cache.resize(n as usize, 0);
        }
        cache[n as usize - 1] = result;
    }
    result
}

fn fib_st_linear(mut n: u64) -> u64 {
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
struct Matrix2x2(u64, u64, u64, u64);

impl Mul for Matrix2x2 {
    type Output = Matrix2x2;
    fn mul(self, rhs: Self) -> Self::Output {
        let a = self.0 * rhs.0 + self.1 * rhs.2;
        let b = self.0 * rhs.1 + self.1 * rhs.3;
        let c = self.2 * rhs.0 + self.3 * rhs.2;
        let d = self.2 * rhs.1 + self.3 * rhs.3;
        Matrix2x2(a, b, c, d)
    }
}

fn fib_st_matrix(mut n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let step = Matrix2x2(0, 1, 1, 1);
    let mut fib = Matrix2x2(0, 1, 1, 1);
    while n > 0 {
        n -= 1;
        fib = fib * step;
    }
    fib.0
}

fn fib_st_matrix_expo(mut n: u64) -> u64 {
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

/// credit to [froststare](https://github.com/camblomquist)
fn fib_st_successor(n: u64) -> u64 {
    std::iter::successors(Some([0, 1]), |&[n_0, n_1]| Some([n_1, n_0 + n_1]))
        .flatten()
        .step_by(2)
        .nth(n as usize)
        .unwrap()
}

#[cfg(test)]
mod tests {
    use crate::{
        fib_linear_gpu_wrapper, fib_matrix_expo_gpu_wrapper, fib_matrix_gpu_wrapper, fib_st_linear,
        fib_st_matrix, fib_st_matrix_expo, fib_st_memo_hashmap, fib_st_memo_vec, fib_st_normal,
        fib_st_successor,
    };

    const FIB: u64 = 40;
    const TARGET_VALUE: u64 = 102334155;

    #[test]
    fn test_normal() {
        assert_eq!(fib_st_normal(FIB), TARGET_VALUE)
    }

    #[test]
    fn test_memo() {
        assert_eq!(fib_st_memo_hashmap(FIB), TARGET_VALUE);
        assert_eq!(fib_st_memo_vec(FIB), TARGET_VALUE);
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

    // this test can hang and crash your gpu
    // #[test]
    // fn test_gpu_normal() {
    //     assert_eq!(fib_normal_gpu_wrapper()(FIB), TARGET_VALUE)
    // }

    #[test]
    fn test_gpu_linear() {
        assert_eq!(fib_linear_gpu_wrapper()(FIB), TARGET_VALUE)
    }

    #[test]
    fn test_gpu_matrix() {
        assert_eq!(fib_matrix_gpu_wrapper()(FIB), TARGET_VALUE)
    }

    #[test]
    fn test_gpu_matrix_expo() {
        assert_eq!(fib_matrix_expo_gpu_wrapper()(FIB), TARGET_VALUE)
    }

    #[test]
    fn test_froststare() {
        assert_eq!(fib_st_successor(FIB), TARGET_VALUE)
    }
}
