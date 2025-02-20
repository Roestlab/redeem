use ndarray::{Array1, Array2};
use plotly::common::{DashType, Line, Mode};
use plotly::{Plot, Histogram, Scatter};
use plotly::layout::{Layout, Axis};
use itertools_num::linspace;

/// Plot a histogram of the scores for the targets and decoys
pub fn plot_score_histogram(scores: &Vec<f64>, labels: &Vec<i32>, title: &str, x_title: &str) -> Result<Plot, String> {
    assert_eq!(scores.len(), labels.len(), "Scores and labels must have the same length");
    assert!(labels.iter().all(|&l| l == 1 || l == -1), "Labels must be 1 for targets and -1 for decoys");

    let mut scores_target = Vec::new();
    let mut scores_decoy = Vec::new();

    for (score, label) in scores.iter().zip(labels.iter()) {
        if *label == 1 {
            scores_target.push(*score);
        } else {
            scores_decoy.push(*score);
        }
    }

    let trace_target = Histogram::new(scores_target).name("Target");
    let trace_decoy = Histogram::new(scores_decoy).name("Decoy");

    let layout = Layout::new()
        .title(title)
        .x_axis(plotly::layout::Axis::new().title(x_title))
        .y_axis(plotly::layout::Axis::new().title("Density"));

    let mut plot = Plot::new();
    plot.add_trace(trace_target);
    plot.add_trace(trace_decoy);
    plot.set_layout(layout);

    Ok(plot)
}



fn ecdf(data: &mut Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = data.len() as f64;
    let y: Vec<f64> = (1..=data.len()).map(|i| i as f64 / n).collect();
    (data.clone(), y)
}

fn interpolate_ecdf(x: &Vec<f64>, y: &Vec<f64>, x_seq: &Vec<f64>) -> Vec<f64> {
    x_seq.iter().map(|&xi| {
        let idx = x.iter().position(|&xv| xv >= xi).unwrap_or(x.len() - 1);
        y[idx]
    }).collect()
}

fn estimate_pi0(decoy_scores: &Vec<f64>, lambda: f64) -> f64 {
    let n = decoy_scores.len() as f64;
    let count_above_lambda = decoy_scores.iter().filter(|&&s| s > lambda).count() as f64;
    count_above_lambda / ((1.0 - lambda) * n)
}

/// Generate a P-P plot as described in Debrie, E. et. al. (2023) Journal of Proteome Research.
/// 
/// # Arguments
/// 
/// * `top_targets` - The top scores for the targets
/// * `top_decoys` - The top scores for the decoys
/// * `title` - The title of the plot
/// 
pub fn plot_pp(scores: &Vec<f64>, labels: &Vec<i32>, title: &str) -> Result<Plot, String> {
    assert_eq!(scores.len(), labels.len(), "Scores and labels must have the same length");
    assert!(labels.iter().all(|&l| l == 1 || l == -1), "Labels must be 1 for targets and -1 for decoys");

    let mut scores_target = Vec::new();
    let mut scores_decoy = Vec::new();

    for (score, label) in scores.iter().zip(labels.iter()) {
        if *label == 1 {
            scores_target.push(*score);
        } else {
            scores_decoy.push(*score);
        }
    }

    let (x_target, y_target) = ecdf(&mut scores_target);
    let (x_decoy, y_decoy) = ecdf(&mut scores_decoy);

    let x_min = x_target.first().unwrap().min(*x_decoy.first().unwrap());
    let x_max = x_target.last().unwrap().max(*x_decoy.last().unwrap());
    let x_seq: Vec<f64> = linspace(x_min, x_max, 1000).collect();

    let y_target_interp = interpolate_ecdf(&x_target, &y_target, &x_seq);
    let y_decoy_interp = interpolate_ecdf(&x_decoy, &y_decoy, &x_seq);

    let pi0 = estimate_pi0(&scores_decoy, 0.5);
    let pi0_line_y: Vec<f64> = y_decoy_interp.iter().map(|&x| pi0 * x).collect();

    let mut plot = Plot::new();

    let scatter = Scatter::new(y_decoy_interp.clone(), y_target_interp.clone())
        .mode(Mode::Markers)
        .name("Target vs Decoy ECDF");

    let reference_line = Scatter::new(vec![0.0, 1.0], vec![0.0, 1.0])
        .mode(Mode::Lines)
        .name("y = x (Perfect match)")
        .line(Line::new().color("red").dash(DashType::Dash));

    let pi0_line = Scatter::new(y_decoy_interp.clone(), pi0_line_y)
        .mode(Mode::Lines)
        .name(format!("Estimated π₀ = {:.3}", pi0))
        .line(Line::new().color("blue").dash(DashType::Dot));

    plot.add_trace(scatter);
    plot.add_trace(reference_line);
    plot.add_trace(pi0_line);
    plot.set_layout(
        Layout::new()
            .title(title)
            .x_axis(plotly::layout::Axis::new().title("Decoy ECDF"))
            .y_axis(plotly::layout::Axis::new().title("Target ECDF")),
    );

    Ok(plot)
}
