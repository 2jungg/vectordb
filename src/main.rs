mod F;
mod tools;

use tools::convert::{TextProcessor};
use F::utils::{cosine_similarity};

fn main() -> anyhow::Result<()> {
    let mut processor = TextProcessor::new("bge-m3/onnx/tokenizer.json", "bge-m3/onnx/model.onnx")?;
    let text = "제 이름은 이중권 입니다!";
    let text2 = "My name is Albert Einstein";

    let output = processor.get_embedding(text)?;
    let output2 = processor.get_embedding(text2)?;

    println!("{:?}", cosine_similarity(&output, &output2));

    Ok(())
}