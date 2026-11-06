//! Builds HiGHS (external/HiGHS) as a static library, as highs-sys does.
//!
//! With HIGHS_BUILD_DIR set (the Makefile sets it), the library is built once per HiGHS version in
//! HIGHS_BUILD_DIR/<version>, for all builds that set it: the PGO builds and the builds of other
//! commits. Otherwise in OUT_DIR. (HiGHS is C++: RUSTFLAGS do not change its build.)

use std::path::{Path, PathBuf};

/// Written after a complete build in HIGHS_BUILD_DIR/<version>
const BUILT_MARKER: &str = "built";

fn build(src: &Path, out_dir: Option<&Path>) -> PathBuf {
    let mut config = cmake::Config::new(src);
    if let Some(dir) = out_dir {
        config.out_dir(dir);
    }
    config
        .profile("Release")
        .define("FAST_BUILD", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTING", "OFF")
        .define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "FALSE")
        .define("ZLIB", "OFF")
        .build()
}

fn main() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../external/HiGHS");
    let version_file = src.join("Version.txt");
    println!("cargo:rerun-if-changed={}", version_file.display());
    println!("cargo:rerun-if-env-changed=HIGHS_BUILD_DIR");

    let dst = match std::env::var_os("HIGHS_BUILD_DIR") {
        Some(build_dir) => {
            // (the version: Version.txt's "HIGHS_MAJOR=1" lines, e.g. "1.12.0")
            let version_text = std::fs::read_to_string(&version_file).expect("external/HiGHS/Version.txt");
            let version: Vec<&str> = ["HIGHS_MAJOR=", "HIGHS_MINOR=", "HIGHS_PATCH="]
                .iter()
                .map(|key| {
                    version_text
                        .lines()
                        .find_map(|line| line.trim().strip_prefix(key))
                        .unwrap_or_else(|| panic!("{key} not in {}", version_file.display()))
                })
                .collect();
            let dir = PathBuf::from(build_dir).join(version.join("."));
            if !dir.join(BUILT_MARKER).exists() {
                build(&src, Some(&dir));
                std::fs::write(dir.join(BUILT_MARKER), "").expect("HIGHS_BUILD_DIR is writable");
            }
            dir
        }
        None => build(&src, None),
    };

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-lib=static=highs");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
