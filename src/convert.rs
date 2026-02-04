use ort::session::{ builder::GraphOptimizationLevel, Session };
use ort::value::TensorRef;
use ndarray::Array2;
use tokenizers::Tokenizer;

#[derive(Debug)]
struct TokenOutput {
    tokens: Vec<String>,
    ids: Vec<u32>,
    attention: Vec<u32>,
}

pub struct TextProcessor {
    tokenizer: Tokenizer,
    model: Session,
}

impl TextProcessor {
    pub fn new(tokenizer_path: &str, model_path: &str) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_execution_providers([
                ort::ep::OpenVINO::default()
                    .with_device_type("GPU")
                    .build()
            ])?
            .commit_from_file(model_path)?;
        
        Ok(Self { tokenizer, model })
    }

    pub fn get_embedding(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        let token_output = self._encode(text)?;
        let token_ids = token_output.ids;
        let attention_mask = token_output.attention;

        let input_shape = (1, token_ids.len());
        let input_tensor = Array2::from_shape_vec(
            input_shape,
            token_ids.iter().map(|&x| x as i64).collect()
        )?;

        let mask_tensor = Array2::from_shape_vec(
            input_shape,
            attention_mask.iter().map(|&x| x as i64).collect()
        )?;

        let outputs = self.model.run(ort::inputs![
            "input_ids" => TensorRef::from_array_view(&input_tensor)?,
            "attention_mask" => TensorRef::from_array_view(&mask_tensor)?
        ])?;

        let output_tensor = outputs["sentence_embedding"].try_extract_array::<f32>()?;
        
        let view = output_tensor.view();
        let embedding = view.slice(ndarray::s![0, ..]).to_vec();

        Ok(embedding)
    }

    fn _encode(&self, text: &str) -> anyhow::Result<TokenOutput> {
        let encoding = self.tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!(e))?;
        Ok(TokenOutput {
            tokens: encoding.get_tokens().to_vec(),
            ids: encoding.get_ids().to_vec(),
            attention: encoding.get_attention_mask().to_vec()
        })
    }

    fn _decode(&self, token_output: &TokenOutput) -> anyhow::Result<String> {
        let decoded_text = self.tokenizer.decode(&token_output.ids, true).map_err(|e| anyhow::anyhow!(e.to_string()))?;

        Ok(decoded_text)
    }
}