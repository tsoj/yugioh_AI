//! Where the data is (ygoai/paths.py): the same environment variables and defaults.

use std::path::PathBuf;

/// The repository (the main one: this crate is crates/ygoai in it), or $YGOAI_ROOT: a build of
/// another version runs with its own lua_fixes/ and the checkout's data (scripts/sprt_branch.sh).
pub fn root() -> PathBuf {
    std::env::var_os("YGOAI_ROOT").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")))
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
    root().join("lua_fixes")
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
