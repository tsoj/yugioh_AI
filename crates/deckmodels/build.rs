//! Embeds the deck models (human_decks/deck_models.bin from export_deck_models.py;
//! $YGO_DECK_MODELS for another file) in the library. Without an export the crate builds without
//! the models (`available()` is false).

use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(deck_models)");
    println!("cargo:rerun-if-env-changed=YGO_DECK_MODELS");
    let file = std::env::var("YGO_DECK_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../human_decks/deck_models.bin"));
    println!("cargo:rerun-if-changed={}", file.display());
    if !file.exists() {
        println!("cargo:warning=deck models not exported ({}): building without them", file.display());
        return;
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::copy(&file, out.join("deck_models.bin")).expect("copy deck_models.bin");
    println!("cargo:rustc-cfg=deck_models");
}
