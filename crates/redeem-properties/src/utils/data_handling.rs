

/// Type of RT normalization used
#[derive(Debug, Clone, Copy)]
pub enum RTNormalization {
    ZScore(f32, f32),     // mean, std
    MinMax(f32, f32),     // min, max
    None,
}

impl RTNormalization {
    pub fn from_str(norm: Option<String>) -> Self {
        match norm.as_deref() {
            Some("z_score") => RTNormalization::ZScore(0.0, 0.0),
            Some("min_max") => RTNormalization::MinMax(0.0, 0.0),
            _ => RTNormalization::None,
        }
    }
}

#[derive(Clone)]
pub struct PeptideData {
    pub sequence: String,
    pub charge: Option<i32>,
    pub precursor_mass: Option<f32>,
    pub nce: Option<i32>,
    pub instrument: Option<String>,
    pub retention_time: Option<f32>,
    pub ion_mobility: Option<f32>,
    pub ccs: Option<f32>,
    pub ms2_intensities: Option<Vec<Vec<f32>>>,
}

impl PeptideData {
    pub fn new(sequence: &str, charge: Option<i32>, precursor_mass: Option<f32>, nce: Option<i32>, instrument: Option<&str>, retention_time: Option<f32>, ion_mobility: Option<f32>, ccs: Option<f32>, ms2_intensities: Option<Vec<Vec<f32>>>) -> Self {
        Self {
            sequence: sequence.to_string(),
            charge,
            precursor_mass,
            nce,
            instrument: instrument.map(|s| s.to_string()),
            retention_time,
            ion_mobility,
            ccs,
            ms2_intensities
        }
    }
}
