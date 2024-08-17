use dashmap::DashMap;
use deepsize::DeepSizeOf;
use flume::bounded;
use lazy_static::lazy_static;
use ocl::ProQue;
use ratatui::crossterm::terminal::disable_raw_mode;
use ratatui::crossterm::terminal::enable_raw_mode;
use ratatui::crossterm::terminal::EnterAlternateScreen;
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
use std::sync::{Arc, Mutex};

lazy_static! {
    static ref PRO_QUE: Arc<Mutex<ProQue>> = Arc::new(Mutex::new(
        ProQue::builder()
            .src(include_str!("../opencl/fibs.cl"))
            .dims(1)
            .build()
            .expect("Failed to build proque")
    ));
    static ref CACHE: std::sync::RwLock<std::collections::HashMap<u64, u64>> =
        std::sync::RwLock::new(std::collections::HashMap::new());
}

fn main() {
    let funcs: HashMap<String, Arc<dyn Fn(u64) -> u64 + Send + Sync>> = {
        let mut map = HashMap::new();
        map.insert(
            "test_singlethreaded_normal".to_string(),
            Arc::new(fib_st_normal) as Arc<dyn Fn(u64) -> u64 + Send + Sync>,
        );
        map.insert(
            "test_singlethreaded_memo".to_string(),
            Arc::new(fib_st_memo) as Arc<dyn Fn(u64) -> u64 + Send + Sync>,
        );
        map.insert(
            "test_singlethreaded_linear".to_string(),
            Arc::new(fib_st_linear) as Arc<dyn Fn(u64) -> u64 + Send + Sync>,
        );
        map.insert(
            "test_singlethreaded_matrix".to_string(),
            Arc::new(fib_st_matrix) as Arc<dyn Fn(u64) -> u64 + Send + Sync>,
        );
        map.insert(
            "test_singlethreaded_matrix_expo".to_string(),
            Arc::new(fib_st_matrix_expo) as Arc<dyn Fn(u64) -> u64 + Send + Sync>,
        );
        // please don't uncomment gpu_normal
        // map.insert("test_gpu_normal".to_string(), Arc::new(fib_normal_gpu_wrapper()) as Arc<dyn Fn(u64) -> u64 + Send + Sync>);
        map.insert(
            "test_gpu_linear".to_string(),
            Arc::new(fib_linear_gpu_wrapper()) as Arc<dyn Fn(u64) -> u64 + Send + Sync>,
        );
        map.insert(
            "test_gpu_matrix".to_string(),
            Arc::new(fib_matrix_gpu_wrapper()) as Arc<dyn Fn(u64) -> u64 + Send + Sync>,
        );
        map.insert(
            "test_gpu_matrix_expo".to_string(),
            Arc::new(fib_matrix_expo_gpu_wrapper()) as Arc<dyn Fn(u64) -> u64 + Send + Sync>,
        );
        map
    };

    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(funcs.len())
    //     .build_global()
    //     .expect("Failed to limit the thread counts");

    let mut h = HashMap::new();
    for (name, func) in funcs.iter() {
        let (sender, receiver) = bounded::<(u64, f64)>(1);
        test(func.clone(), sender);
        h.insert(name.clone(), receiver);
    }
    run_tui(h).unwrap();
}

fn run_tui(
    tests: HashMap<String, flume::Receiver<(u64, f64)>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    let progress: DashMap<&str, (u64, bool, f64)> = DashMap::new();

    enable_raw_mode().unwrap();
    stdout().execute(EnterAlternateScreen).unwrap();

    loop {
        tests.par_iter().for_each(|(name, r)| {
            if let Ok(a) = r.try_recv() {
                if a.0 != 0 {
                    progress
                        .entry(name)
                        .and_modify(|i| {
                            i.0 = a.0;
                            i.2 = a.1;
                        })
                        .or_insert((a.0, false, a.1));
                    return;
                }
                progress.entry(name).and_modify(|i| {
                    i.1 = true;
                    i.2 = a.1;
                });
            }
            if r.is_disconnected() {
                progress.entry(name).and_modify(|i| i.1 = true);
            }
        });

        if progress.par_iter().all(|i| i.value().1) && !progress.is_empty() {
            break;
        }
        terminal
            .draw(|f| {
                draw_ui(f, &progress);
            })
            .unwrap();
        // std::thread::sleep(std::time::Duration::from_millis(50));
    }

    disable_raw_mode().unwrap();

    Ok(())
}

fn draw_ui(frame: &mut Frame, progress: &DashMap<&str, (u64, bool, f64)>) {
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
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![
                    Constraint::Percentage(100 / progress.len() as u16 + 1);
                    progress.len() + 1
                ])
                .split(frame.area());

            for (i, n) in progress.iter().enumerate() {
                let test_name = n.key();
                let prog = n.value();
                let text = format!(
                    "Status: Solving Fibonnaci {} ({})\nSpeed: {} Fibbonaci number/second",
                    prog.0,
                    {
                        if prog.1 {
                            "Done"
                        } else {
                            "Running"
                        }
                    },
                    prog.2
                );

                let paragraph = Paragraph::new(text).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(test_name.to_string())
                        .border_style(match prog.1 {
                            true => Color::Green,
                            false => Color::Cyan,
                        }),
                );

                frame.render_widget(paragraph, chunks[i]);
            }
            let cache_stat = Paragraph::new(format!(
                "CACHE Size is currently: {} megabytes",
                CACHE.deep_size_of() as f32 / 1_048_576_f32
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

fn test(call: Arc<dyn Fn(u64) -> u64 + Send + Sync>, tx: flume::Sender<(u64, f64)>) {
    rayon::spawn(move || {
        let mut x = 1;
        let mut previous_fns = 0.0;
        loop {
            let now = std::time::Instant::now();
            call(x);
            let ex_time = std::time::Instant::now();
            let duration = (ex_time - now).as_secs_f32();
            if duration > 1.0 {
                break;
            }
            previous_fns = 1.0 / duration as f64;
            tx.send((x, previous_fns)).expect("Failed to send data");
            x += 1;
        }
        tx.send((0, previous_fns))
            .expect("Failed to send final progress"); // mark as done
    });
}

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

fn fib_st_memo(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    {
        let cache = CACHE.read().unwrap();
        if let Some(&result) = cache.get(&n) {
            return result;
        }
    }
    let result = fib_st_memo(n - 1) + fib_st_memo(n - 2);
    {
        let mut cache = CACHE.write().unwrap();
        cache.insert(n, result);
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

#[cfg(test)]
mod tests {
    use crate::{
        fib_linear_gpu_wrapper, fib_matrix_expo_gpu_wrapper, fib_matrix_gpu_wrapper, fib_st_linear,
        fib_st_matrix, fib_st_matrix_expo, fib_st_memo, fib_st_normal,
    };

    const FIB: u64 = 40;
    const TARGET_VALUE: u64 = 102334155;

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
}
