use candle_core::{Result, Tensor};

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