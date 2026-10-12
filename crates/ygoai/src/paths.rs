//! Where the data is (ygoai/paths.py): the same environment variables and defaults.

use std::path::PathBuf;

/// The repository (the main one: this crate is crates/ygoai in it).
pub fn root() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn env_or(var: &str, default: PathBuf) -> PathBuf {
    std::env::var_os(var).map(PathBuf::from).unwrap_or(default)
}

/// The card scripts ($YGOAI_SCRIPTS, else external/CardScripts).
pub fn card_scripts_dir() -> PathBuf {
    env_or("YGOAI_SCRIPTS", root().join("external").join("CardScripts"))
}

/// The project's corrected card scripts.
pub fn fixed_card_scripts_dir() -> PathBuf {
    root().join("core_ext").join("lua").join("cards")
}

/// The deck corpus ($YGOAI_DECKS, else download_human_decks/decks).
pub fn decks_dir() -> PathBuf {
    env_or("YGOAI_DECKS", root().join("download_human_decks").join("decks"))
}

/// The newer deck corpus ($YGOAI_NEW_DECKS, else download_human_decks/decks_new).
pub fn new_decks_dir() -> PathBuf {
    env_or("YGOAI_NEW_DECKS", root().join("download_human_decks").join("decks_new"))
}

/// build/ of the repository.
pub fn build_dir() -> PathBuf {
    root().join("build")
}
