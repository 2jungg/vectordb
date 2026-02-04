mod convert;
mod func_utils;

use convert::{TextProcessor};
use func_utils::{cosine_similarity};

fn main() -> anyhow::Result<()> {
    let dylib_path = "C:/Users/hanta/Documents/projects/MyVectorDB/lib/onnxruntime.dll";
    ort::init_from(dylib_path)?.commit();

    let mut processor = TextProcessor::new("bge-m3/onnx/tokenizer.json", "bge-m3/onnx/model.onnx")?;
    let text = "제 이름은 이중권 입니다!";
    let text2 = "My name is Albert Einstein";

    let output = processor.get_embedding(text)?;
    let output2 = processor.get_embedding(text2)?;

    println!("{:?}", cosine_similarity(&output, &output2));

    Ok(())
}