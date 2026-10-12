//! Hashes of the sources the binding is built from, for the game cache (ygoai/game_cache.py): one of
//! everything that plays a game (the engine, the cards, the ygoai crate and this binding), one of
//! the belief modules (only games of agents that use the belief depend on them).

use std::path::{Path, PathBuf};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name();
        if p.is_dir() {
            if name != "target" && name != ".git" {
                files(&p, out);
            }
        } else {
            out.push(p);
        }
    }
}

/// FNV-1a over the files' relative paths and contents (in path order).
fn hash(base: &Path, paths: &[PathBuf]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut eat = |b: &[u8]| {
        for &x in b {
            h ^= x as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    for p in paths {
        eat(p.strip_prefix(base).unwrap_or(p).to_string_lossy().as_bytes());
        eat(&std::fs::read(p).unwrap_or_default());
    }
    h
}

fn main() {
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = here.parent().unwrap().to_path_buf();
    let dirs = [root.join("rust/ygo-core"), root.join("rust/ygo-cards"), root.join("crates/ygoai"), here.clone()];
    let belief = [root.join("crates/ygoai/src/belief"), here.join("src/belief.rs")];
    let mut all = Vec::new();
    for d in &dirs {
        println!("cargo:rerun-if-changed={}", d.display());
        files(d, &mut all);
    }
    all.retain(|p| !p.ends_with("Cargo.lock"));
    all.sort();
    let (bel, eng): (Vec<PathBuf>, Vec<PathBuf>) = all.into_iter().partition(|p| belief.iter().any(|b| p.starts_with(b)));
    println!("cargo:rustc-env=YGOAI_ENGINE_SOURCES={:016x}", hash(&root, &eng));
    println!("cargo:rustc-env=YGOAI_BELIEF_SOURCES={:016x}", hash(&root, &bel));
}
