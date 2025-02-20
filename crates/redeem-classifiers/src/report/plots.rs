use ndarray::{Array1, Array2};
use plotly::common::Mode;
use plotly::{Plot, Histogram, Scatter};
use plotly::layout::{Layout, Axis};
use itertools_num::linspace;

/// Plot a histogram of the scores for the targets and decoys
pub fn plot_score_histogram(scores: &Array1<f32>, labels: &Array1<i32>, title: &str) -> Result<Plot, String> {
    // Assert that the scores and labels have the same length
    assert_eq!(scores.len(), labels.len(), "Scores and labels must have the same length");

    // Assert that the labels are only two classes
    assert!(labels.iter().all(|&l| l == 1 || l == -1), "Labels must be composed of only two classes, 1 for targets and -1 for decoys");

    let mut scores_target = Vec::new();
    let mut scores_decoy = Vec::new();

    for (score, label) in scores.iter().zip(labels.iter()) {
        if *label == 1 {
            scores_target.push(*score);
        } else {
            scores_decoy.push(*score);
        }
    }

    let trace_target = Histogram::new(scores_target)
        .name("Target");

    let trace_decoy = Histogram::new(scores_decoy)
        .name("Decoy");

    let layout = Layout::new()
        .title(title)
        .x_axis(Axis::new().title("Score"))
        .y_axis(Axis::new().title("Density"));

    let mut plot = Plot::new();
    plot.add_trace(trace_target);
    plot.add_trace(trace_decoy);
    plot.set_layout(layout);

    Ok(plot)
}



fn ecdf(data: &mut Vec<f32>) -> (Vec<f32>, Vec<f32>) {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = data.len() as f32;
    let y: Vec<f32> = (1..=data.len()).map(|i| i as f32 / n).collect();
    (data.clone(), y)
}

fn interpolate_ecdf(x: &Vec<f32>, y: &Vec<f32>, x_seq: &Vec<f32>) -> Vec<f32> {
    x_seq.iter().map(|&xi| {
        let idx = x.iter().position(|&xv| xv >= xi).unwrap_or(x.len() - 1);
        y[idx]
    }).collect()
}

/// Generate a P-P plot as described in Debrie, E. et. al. (2023) Journal of Proteome Research.
/// 
/// # Arguments
/// 
/// * `top_targets` - The top scores for the targets
/// * `top_decoys` - The top scores for the decoys
/// * `title` - The title of the plot
/// 
pub fn plot_pp(scores: &Array1<f32>, labels: &Array1<i32>, title: &str) -> Result<Plot, String> {

    // Assert that the scores and labels have the same length
    assert_eq!(scores.len(), labels.len(), "Scores and labels must have the same length");

    // Assert that the labels are only two classes
    assert!(labels.iter().all(|&l| l == 1 || l == -1), "Labels must be composed of only two classes, 1 for targets and -1 for decoys");

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
    let x_seq: Vec<f32> = linspace(x_min, x_max, 1000).collect();

    let y_target_interp = interpolate_ecdf(&x_target, &y_target, &x_seq);
    let y_decoy_interp = interpolate_ecdf(&x_decoy, &y_decoy, &x_seq);

    let mut plot = Plot::new();

    let scatter = Scatter::new(y_decoy_interp.clone(), y_target_interp.clone())
        .mode(plotly::common::Mode::Markers)
        .name("Target vs Decoy ECDF");
    
    let reference_line = Scatter::new(vec![0.0, 1.0], vec![0.0, 1.0])
        .mode(plotly::common::Mode::Lines)
        .name("y = x (Perfect match)")
        .line(plotly::common::Line::new().color("red").dash(plotly::common::DashType::Dash));
    
    plot.add_trace(scatter);
    plot.add_trace(reference_line);
    plot.set_layout(Layout::new().title(title).x_axis(plotly::layout::Axis::new().title("Decoy ECDF")).y_axis(plotly::layout::Axis::new().title("Target ECDF")));
    
    Ok(plot)
}
