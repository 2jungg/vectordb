mod utils;
mod tools;
mod types;

use tools::embedder::Embedder;
use utils::utils::cosine_similarity;

fn main() -> anyhow::Result<()> {
    let mut embedder = Embedder::new("bge-m3/onnx/tokenizer.json", "bge-m3/onnx/model.onnx")?;
    let text = "제 이름은 이중권 입니다!";
    let text2 = "My name is Albert Einstein";

    let output = embedder.get_embedding(text)?;
    let output2 = embedder.get_embedding(text2)?;

    println!("Cosine Similarity: {:?}", cosine_similarity(&output, &output2));

    Ok(())
}