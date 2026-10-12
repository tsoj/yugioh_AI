//! Builds HiGHS (external/HiGHS) as a static library, as highs-sys does.

fn main() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../external/HiGHS");
    let dst = cmake::Config::new(&src)
        .profile("Release")
        .define("FAST_BUILD", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTING", "OFF")
        .define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "FALSE")
        .define("ZLIB", "OFF")
        .build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-lib=static=highs");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed={}", src.join("Version.txt").display());
}
