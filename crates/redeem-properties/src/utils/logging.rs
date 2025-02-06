use candle_core::{Result, Tensor};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ops::Range;
use tqdm::Tqdm;
use tqdm::tqdm;

pub struct Progress {
    total: usize,
    progress: Arc<Mutex<Tqdm<Range<usize>>>>, 
    count: AtomicUsize,               // Atomic counter for tracking progress
    sender: mpsc::Sender<usize>,      // Channel to send updates
    progress_thread: Option<thread::JoinHandle<()>>, // Background thread to update tqdm
    description: String,              // Description for the progress bar
}

impl Progress {
    pub fn new(total: usize, description: &str) -> Self {
        let progress = Arc::new(Mutex::new(tqdm(0..total).desc(Some(description)))); // Initialize Tqdm
        let count = AtomicUsize::new(0);

        let (tx, rx) = mpsc::channel();
        let progress_clone = Arc::clone(&progress);

        // Spawn a thread to handle progress updates
        let handle = thread::spawn(move || {
            for _ in rx {
                let _ = progress_clone.lock().unwrap().pbar.update(1); // Always update by 1
            }
        });

        Self {
            total,
            progress,
            count,
            sender: tx,
            progress_thread: Some(handle),
            description: description.to_string(),
        }
    }

    /// Increment progress counter safely using a channel
    pub fn inc(&self) {
        let new_count = self.count.fetch_add(1, Ordering::AcqRel) + 1;
    
        if new_count > self.total {
            println!("⚠️ WARNING: Extra update detected! Skipping...");
            return; // Prevent overflow
        }

        let _ = self.sender.send(1); // Always send 1 instead of new_count
    }

    /// Ensure progress updates are completed before exiting
    pub fn finish(self) {
        drop(self.sender); // Close the channel to signal the thread to finish
        if let Some(handle) = self.progress_thread {
            let _ = handle.join(); // Wait for the progress thread to exit
        }
    }
}

pub fn print_tensor(
    tensor: &Tensor,
    decimal_places: usize,
    max_rows: Option<usize>,
    max_cols: Option<usize>,
) -> Result<()> {
    let shape = tensor.shape().dims();
    if shape.len() != 3 {
        return Err(candle_core::Error::Msg("Expected a 3D tensor".to_string()));
    }

    let (batch, seq_len, features) = (shape[0], shape[1], shape[2]);
    let data = tensor.to_vec3::<f32>()?;

    let max_cols = max_cols.unwrap_or(features);

    println!("tensor([");
    for b in 0..batch {
        println!("  [");
        let rows_to_print = max_rows.unwrap_or(seq_len).min(seq_len);
        for s in 0..rows_to_print {
            print!("    [");
            let cols_to_print = max_cols.min(features);
            if cols_to_print * 2 < features {
                for f in 0..cols_to_print {
                    print!("{:.*e}", decimal_places, data[b][s][f]);
                    if f < cols_to_print - 1 {
                        print!(", ");
                    }
                }
                print!(", ..., ");
                for f in (features - cols_to_print)..features {
                    print!("{:.*e}", decimal_places, data[b][s][f]);
                    if f < features - 1 {
                        print!(", ");
                    }
                }
            } else {
                for f in 0..features {
                    print!("{:.*e}", decimal_places, data[b][s][f]);
                    if f < features - 1 {
                        print!(", ");
                    }
                }
            }
            if s < rows_to_print - 1 {
                println!("],");
            } else {
                print!("]");
            }
        }
        if rows_to_print < seq_len {
            println!(",");
            println!("    ...");
        }
        println!("  ],");
    }
    println!("])");

    Ok(())
}


#[cfg(test)]
mod tests {
    use rayon::prelude::*;
    use super::*;

    #[test]
    fn test_progress_it(){
        let total = 24;
        let pbar = Progress::new(total, "Testing Rayon");
        let v = vec![1; total];
        let mapped: Vec<i32> = v
            .into_par_iter()
            .enumerate()
            .map(|(_i, x)| {
                pbar.inc();
                x * 100
            })
            .collect();
        pbar.finish();
        println!("{:?}", mapped);
    }
}