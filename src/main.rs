mod convert;
mod func_utils;

use std:: {env, path::PathBuf};
use convert::{TextProcessor};
use func_utils::{cosine_similarity};

fn get_dll_path(dll_name: &str) -> PathBuf {
    let mut exe_path = env::current_exe().expect("Failed to get exe path");
    exe_path.pop();
    exe_path.push(dll_name);
    exe_path
}
fn main() -> anyhow::Result<()> {
    let dylib_path = get_dll_path("onnxruntime.dll");
    println!("Attempting to load ONNX Runtime DLL from: {:?}", dylib_path);
    ort::init_from(dylib_path)?.commit();

    let mut processor = TextProcessor::new("bge-m3/onnx/tokenizer.json", "bge-m3/onnx/model.onnx")?;
    let text = "제 이름은 이중권 입니다!";
    let text2 = "My name is Albert Einstein";

    let output = processor.get_embedding(text)?;
    let output2 = processor.get_embedding(text2)?;

    println!("{:?}", cosine_similarity(&output, &output2));

    Ok(())
}