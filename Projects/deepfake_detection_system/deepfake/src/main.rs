mod fft;

use std::fs;
use std::path::Path;
use image::GenericImageView;

fn process_dir(base: &str, split: &str, label: &str) {
    let dir = format!("{}/{}/{}", base, split, label);
    let path = Path::new(&dir);

    println!("\n[{} / {}]", split.to_uppercase(), label.to_uppercase());

    let entries = fs::read_dir(path)
        .expect("Failed to read directory");

    for entry in entries {
        let entry = entry.expect("Invalid entry");
        let file_path = entry.path();

        if file_path.extension().and_then(|s| s.to_str()) != Some("jpg") {
            continue;
        }

        let img = image::open(&file_path)
            .expect("Failed to open image")
            .to_luma8();

        let score = fft::frequency_score(&img);

        println!(
            "{:<20}  score = {:.4}",
            file_path.file_name().unwrap().to_string_lossy(),
            score
        );
    }
}

fn main() {
    let dataset_root = "dataset";

    process_dir(dataset_root, "train", "fake");
    process_dir(dataset_root, "train", "real");
    process_dir(dataset_root, "test", "fake");
    process_dir(dataset_root, "test", "real");
}
