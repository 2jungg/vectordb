use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let _profile = env::var("PROFILE").unwrap();
    
    let project_dir_str = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_dir = PathBuf::from(&project_dir_str);

    let dll_source = project_dir.join("lib").join("onnxruntime.dll");

    let out_dir = env::var("OUT_DIR").unwrap();

    let target_dir = PathBuf::from(out_dir)
        .ancestors()
        .nth(3) 
        .unwrap()
        .to_path_buf();

    let dest_path_dll = target_dir.join("onnxruntime.dll");
    let dest_path_so = target_dir.join("libonnxruntime.so");

    if dll_source.exists() {
        fs::copy(&dll_source, &dest_path_dll).expect("DLL COPY FAILED");
        println!("cargo:warning=Copied ONNX Runtime DLL from {:?} to {:?}", dll_source, dest_path_dll);
    }

    // Check for .so file and copy it if it exists (for Linux)
    let so_source = project_dir.join("lib").join("libonnxruntime.so");
    if so_source.exists() {
        fs::copy(&so_source, &dest_path_so).expect("SO COPY FAILED");
        println!("cargo:warning=Copied ONNX Runtime SO from {:?} to {:?}", so_source, dest_path_so);
    } else {
        println!("cargo:warning=ONNX Runtime DLL/SO not found at {:?} or {:?}", dll_source, so_source);
    }

    println!("cargo:rerun-if-changed=lib/onnxruntime.dll");
    println!("cargo:rerun-if-changed=lib/libonnxruntime.so");
}