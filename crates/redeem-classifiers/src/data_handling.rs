use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, SeedableRng};


#[derive(Debug, Clone)]
pub struct Experiment {
    pub x: Array2<f32>,
    pub y: Array1<i32>,
    pub is_train: Array1<bool>,
    pub is_top_peak: Array1<bool>,
    pub tg_num_id: Array1<i32>,
    pub classifier_score: Array1<f32>,
}

impl Experiment {
    pub fn new(x: Array2<f32>, y: Array1<i32>) -> Self {
        let n_samples = x.nrows();
        Experiment {
            x,
            y,
            is_train: Array1::from_elem(n_samples, false),
            is_top_peak: Array1::from_elem(n_samples, false),
            tg_num_id: Array1::from_elem(n_samples, 0),
            classifier_score: Array1::from_elem(n_samples, 0.0),
        }
    }

    pub fn log_summary(&self) {
        println!("Info: Summary of input data:");
        println!("Info: {} peak groups", self.x.nrows());
        println!("Info: {} scores including main score", self.x.ncols());
    }

    pub fn set_and_rerank(&mut self, scores: Array1<f32>) {
        self.classifier_score = scores;
        self.rank_by();
    }

    pub fn rank_by(&mut self) {
        // Implement ranking logic here
        // This is a placeholder and should be replaced with actual ranking logic
        self.is_top_peak = Array1::from_elem(self.x.nrows(), true);
    }

    pub fn get_top_test_peaks(&self) -> Experiment {
        let mask = &self.is_train.mapv(|x| !x) & &self.is_top_peak;
        self.filter(&mask)
    }
    

    pub fn get_decoy_peaks(&self) -> Experiment {
        // Sage represents decoy peaks as -1
        let mask = &self.y.mapv(|v| v == -1);  
        self.filter(mask)
    }

    pub fn get_target_peaks(&self) -> Experiment {
        // Sage represents target peaks as 1
        let mask = &self.y.mapv(|v| v != 1);  
        self.filter(mask)
    }

    pub fn get_top_decoy_peaks(&self) -> Experiment {
        let mask = &self.y.mapv(|v| v == 0) & &self.is_top_peak;
        self.filter(&mask)
    }
    
    pub fn get_top_target_peaks(&self) -> Experiment {
        let mask = &self.y.mapv(|v| v != 0) & &self.is_top_peak;
        self.filter(&mask)
    }

    pub fn get_feature_matrix(&self) -> Array2<f32> {
        self.x.clone()
    }

    pub fn filter(&self, mask: &Array1<bool>) -> Experiment {
        Experiment {
            x: self.x.select(Axis(0), &mask.iter().enumerate().filter_map(|(i, &m)| if m { Some(i) } else { None }).collect::<Vec<_>>()),
            y: self.y.select(Axis(0), &mask.iter().enumerate().filter_map(|(i, &m)| if m { Some(i) } else { None }).collect::<Vec<_>>()),
            is_train: self.is_train.select(Axis(0), &mask.iter().enumerate().filter_map(|(i, &m)| if m { Some(i) } else { None }).collect::<Vec<_>>()),
            is_top_peak: self.is_top_peak.select(Axis(0), &mask.iter().enumerate().filter_map(|(i, &m)| if m { Some(i) } else { None }).collect::<Vec<_>>()),
            tg_num_id: self.tg_num_id.select(Axis(0), &mask.iter().enumerate().filter_map(|(i, &m)| if m { Some(i) } else { None }).collect::<Vec<_>>()),
            classifier_score: self.classifier_score.select(Axis(0), &mask.iter().enumerate().filter_map(|(i, &m)| if m { Some(i) } else { None }).collect::<Vec<_>>()),
        }
    }

    pub fn add_peak_group_rank(&mut self) {
        // Implement peak group ranking logic here
        // This is a placeholder and should be replaced with actual ranking logic
    }

    pub fn split_for_xval(&mut self, fraction: f32, is_test: bool) {
        let mut rng = thread_rng();
        let n_samples = self.x.nrows();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        
        if !is_test {
            indices.shuffle(&mut rng);
        } else {
            indices.sort_unstable();
        }

        let n_train = (n_samples as f32 * fraction) as usize;
        for (i, &idx) in indices.iter().enumerate() {
            self.is_train[idx] = i < n_train;
        }
    }

    pub fn get_train_peaks(&self) -> Experiment {
        let mask = &self.is_train;  
        self.filter(mask)
    }
}



pub struct DataHandler {
    n_folds: usize,
}

impl DataHandler {
    pub fn new(n_folds: usize) -> Self {
        DataHandler { n_folds }
    }

    pub fn create_folds(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut rng = thread_rng();
        let fold_size = n_samples / self.n_folds;
        let mut fold_indices: Vec<usize> = (0..n_samples).collect();
        fold_indices.shuffle(&mut rng);

        (0..self.n_folds)
            .map(|fold| {
                let start = fold * fold_size;
                let end = if fold == self.n_folds - 1 {
                    n_samples
                } else {
                    (fold + 1) * fold_size
                };

                let test_indices: Vec<usize> = fold_indices[start..end].to_vec();
                let train_indices: Vec<usize> = fold_indices
                    .iter()
                    .filter(|&&i| i < start || i >= end)
                    .cloned()
                    .collect();

                (train_indices, test_indices)
            })
            .collect()
    }

    pub fn select_data<'a>(
        &self,
        x: &'a ArrayView2<f32>,
        y: &'a [i32],
        indices: &[usize],
    ) -> (Array2<f32>, Vec<i32>) {
        let x_selected = x.select(Axis(0), indices);
        let y_selected: Vec<i32> = indices.iter().map(|&i| y[i]).collect();
        (x_selected, y_selected)
    }
    
    // Function to split data into training and testing sets
    pub fn train_test_split(
        &self,
        x: &Array2<f32>,
        y: &Array1<i32>,
        test_size: Option<f32>,
        random_state: Option<u64>,
        shuffle: bool,
    ) -> (Array2<f32>, Array1<i32>, Array2<f32>, Array1<i32>) {
        let n_samples = x.nrows();
        let mut rng = thread_rng();

        // Generate indices for shuffling
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Shuffle if required
        if shuffle {
            if let Some(seed) = random_state {
                let mut std_rng = StdRng::seed_from_u64(seed);
                indices.shuffle(&mut std_rng);
            } else {
                indices.shuffle(&mut rng);
            }
        }

        // Determine split sizes
        let test_size = test_size.unwrap_or(0.25); // Default to 25% if not specified
        let n_test = (n_samples as f32 * test_size).round() as usize;
        let n_train = n_samples - n_test;

        // Split indices into train and test
        let train_indices = &indices[..n_train];
        let test_indices = &indices[n_train..];

        // Select training and testing data
        let x_train = x.select(ndarray::Axis(0), train_indices);
        let y_train = y.select(ndarray::Axis(0), train_indices);
        let x_test = x.select(ndarray::Axis(0), test_indices);
        let y_test = y.select(ndarray::Axis(0), test_indices);

        (x_train.to_owned(), y_train.to_owned(), x_test.to_owned(), y_test.to_owned())
    }
    
}
