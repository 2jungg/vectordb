use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let profile = env::var("PROFILE").unwrap();
    
    let project_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let dll_source = PathBuf::from(project_dir).join("lib").join("onnxruntime.dll");

    let out_dir = env::var("OUT_DIR").unwrap();

    let target_dir = PathBuf::from(out_dir)
        .ancestors()
        .nth(3) 
        .unwrap()
        .to_path_buf();

    let dest_path = target_dir.join("onnxruntime.dll");
    
    if dll_source.exists() {
        fs::copy(&dll_source, &dest_path).expect("DLL COPY FAILED");
        println!("cargo:warning=Copied ONNX Runtime DLL from {:?} to {:?}", dll_source, dest_path);
    } else {
        println!("cargo:warning=ONNX Runtime DLL not found at {:?}", dll_source);
    }

    println!("cargo:rerun-if-changed=lib/onnxruntime.dll");
}