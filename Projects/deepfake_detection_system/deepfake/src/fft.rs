use image::GrayImage;
use num_complex::Complex;
use rustfft::FftPlanner;

pub fn frequency_score(img: &GrayImage) -> f64 {
    let (w, h) = img.dimensions();
    let size = (w * h) as usize;

    // Convert pixels to complex signal
    let mut buffer: Vec<Complex<f64>> = img
        .pixels()
        .map(|p| Complex::new(p[0] as f64 / 255.0, 0.0))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(size);
    fft.process(&mut buffer);

    let mut low_energy = 0.0;
    let mut high_energy = 0.0;

    for (i, c) in buffer.iter().enumerate() {
        let mag = c.norm();
        if i < size / 4 {
            low_energy += mag;
        } else {
            high_energy += mag;
        }
    }

    // High-frequency dominance ratio
    high_energy / (low_energy + 1e-9)
}
