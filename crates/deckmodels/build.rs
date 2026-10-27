//! Generates the models' burn code from their ONNX export (human_decks/export_deck_models.py;
//! $YGOAI_DECK_MODELS_ONNX for another directory), with the weights and the vocabulary
//! (deck_models.json) embedded in the library. Without
//! an export the crate builds without the models (`available()` is false).

use std::path::PathBuf;

use burn_onnx::{LoadStrategy, ModelGen};

const MODELS: [&str; 2] = ["deck_existence", "deck_counts"];

fn main() {
    println!("cargo::rustc-check-cfg=cfg(deck_models)");
    println!("cargo:rerun-if-env-changed=YGOAI_DECK_MODELS_ONNX");
    let dir = std::env::var("YGOAI_DECK_MODELS_ONNX").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../human_decks/onnx_export")
    });
    let mut files: Vec<PathBuf> = MODELS.iter().map(|m| dir.join(format!("{m}.onnx"))).collect();
    files.push(dir.join("deck_models.json"));
    for f in &files {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    if !files.iter().all(|f| f.exists()) {
        println!("cargo:warning=deck models not exported ({}): building without them", dir.display());
        return;
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::copy(&files[2], out.join("deck_models.json")).expect("copy deck_models.json");
    for f in &files[..2] {
        ModelGen::new().input(f.to_str().unwrap()).out_dir("model/").load_strategy(LoadStrategy::Embedded).run_from_script();
    }
    println!("cargo:rustc-cfg=deck_models");
}
