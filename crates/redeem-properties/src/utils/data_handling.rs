

pub struct PeptideData {
    pub sequence: String,
    pub charge: Option<i32>,
    pub nce: Option<i32>,
    pub instrument: Option<String>,
    pub retention_time: Option<f32>,
    pub ion_mobility: Option<f32>,
    pub ms2_intensities: Option<Vec<Vec<f32>>>,
}

impl PeptideData {
    pub fn new(sequence: &str, charge: Option<i32>, nce: Option<i32>, instrument: Option<&str>, retention_time: Option<f32>, ion_mobility: Option<f32>, ms2_intensities: Option<Vec<Vec<f32>>>) -> Self {
        Self {
            sequence: sequence.to_string(),
            charge,
            nce,
            instrument: instrument.map(|s| s.to_string()),
            retention_time,
            ion_mobility,
            ms2_intensities
        }
    }
}
