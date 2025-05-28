use plotly::{Layout, Plot, Scatter};
use plotly::common::{Fill, Mode, Title};
use redeem_properties::utils::stats::{TrainingStepMetrics, TrainingPhase};


pub fn plot_losses(
    epoch_losses: &[(usize, f32, Option<f32>, f32, Option<f32>)]
) -> Plot {
    let epochs: Vec<_> = epoch_losses.iter().map(|(e, _, _, _, _)| *e as f64).collect();

    let train_mean: Vec<_> = epoch_losses.iter().map(|(_, m, _, _, _)| *m as f64).collect();
    let train_std: Vec<_> = epoch_losses.iter().map(|(_, _, _, std, _)| *std as f64).collect();
    let train_upper: Vec<_> = train_mean.iter().zip(&train_std).map(|(m, s)| m + s).collect();
    let train_lower: Vec<_> = train_mean.iter().zip(&train_std).map(|(m, s)| m - s).collect();

    let val_mean: Vec<_> = epoch_losses.iter().map(|(_, _, val, _, _)| val.unwrap_or(f32::NAN) as f64).collect();
    let val_std: Vec<_> = epoch_losses.iter().map(|(_, _, _, _, val_std)| val_std.unwrap_or(0.0) as f64).collect();
    let val_upper: Vec<_> = val_mean.iter().zip(&val_std).map(|(m, s)| m + s).collect();
    let val_lower: Vec<_> = val_mean.iter().zip(&val_std).map(|(m, s)| m - s).collect();

    let mut plot = Plot::new();

    // Training loss line
    plot.add_trace(
        Scatter::new(epochs.clone(), train_mean.clone())
            .name("Train Loss")
            .mode(Mode::Lines)
            .line(plotly::common::Line::new().color("rgba(31, 119, 180, 1.0)")),
    );

    // Training loss band
    let mut train_band_y = train_upper.clone();
    let mut train_band_x = epochs.clone();
    let mut lower_reversed: Vec<_> = train_lower.iter().cloned().rev().collect();
    let mut x_reversed: Vec<_> = epochs.iter().cloned().rev().collect();
    train_band_y.extend(lower_reversed);
    train_band_x.extend(x_reversed);

    plot.add_trace(
        Scatter::new(train_band_x, train_band_y)
            .name("Train ± σ")
            .mode(Mode::Lines)
            .fill(Fill::ToSelf)
            .line(plotly::common::Line::new().width(0.0))
            .fill_color("rgba(31, 119, 180, 0.2)")
    );

    // Validation loss line
    plot.add_trace(
        Scatter::new(epochs.clone(), val_mean.clone())
            .name("Val Loss")
            .mode(Mode::Lines)
            .line(plotly::common::Line::new().color("rgba(255, 127, 14, 1.0)")),
    );

    // Validation loss band
    let mut val_band_y = val_upper.clone();
    let mut val_band_x = epochs.clone();
    let mut val_lower_rev: Vec<_> = val_lower.iter().cloned().rev().collect();
    let mut val_x_rev: Vec<_> = epochs.iter().cloned().rev().collect();
    val_band_y.extend(val_lower_rev);
    val_band_x.extend(val_x_rev);

    plot.add_trace(
        Scatter::new(val_band_x, val_band_y)
            .name("Val ± σ")
            .mode(Mode::Lines)
            .fill(Fill::ToSelf)
            .line(plotly::common::Line::new().width(0.0))
            .fill_color("rgba(255, 127, 14, 0.2)")
    );

    plot.set_layout(
        Layout::new()
            .title("Training and Validation Loss Over Epochs")
            .x_axis(plotly::layout::Axis::new().title("Epoch"))
            .y_axis(plotly::layout::Axis::new().title("Loss"))
    );

    plot
}



/// Plot a single training metric (e.g. loss, learning rate, accuracy) over steps.
pub fn plot_training_metric(
    metrics: &TrainingStepMetrics,
    metric_name: &str,
    title: &str,
    x_title: &str,
    y_title: &str,
) -> Plot {
    let mut plot = Plot::new();

    let mut train_x = vec![];
    let mut train_y = vec![];
    let mut val_x = vec![];
    let mut val_y = vec![];

    for i in 0..metrics.steps.len() {
        let x = metrics.steps[i] as f64;
        let y_opt = match metric_name {
            "loss" => Some(metrics.losses[i] as f64),
            "lr" => Some(metrics.learning_rates[i]),
            "accuracy" => metrics.accuracies[i].map(|a| a as f64),
            _ => None,
        };

        if let Some(y) = y_opt {
            match metrics.phases[i] {
                TrainingPhase::Train => {
                    train_x.push(x);
                    train_y.push(y);
                }
                TrainingPhase::Validation => {
                    val_x.push(x);
                    val_y.push(y);
                }
            }
        }
    }

    if !train_x.is_empty() {
        plot.add_trace(
            Scatter::new(train_x.clone(), train_y.clone())
                .mode(Mode::Lines)
                .name("Train"),
        );
    }
    if !val_x.is_empty() {
        plot.add_trace(
            Scatter::new(val_x.clone(), val_y.clone())
                .mode(Mode::Lines)
                .name("Validation"),
        );
    }

    plot.set_layout(
        Layout::new()
            .title(title)
            .x_axis(plotly::layout::Axis::new().title(x_title))
            .y_axis(plotly::layout::Axis::new().title(y_title))
    );

    plot
}
