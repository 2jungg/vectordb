use ort::session::{ builder::GraphOptimizationLevel, Session };
use ort::value::TensorRef;
use ndarray::Array2;
use tokenizers::Tokenizer;
use std:: {env, path::PathBuf};

#[derive(Debug)]
pub struct TokenOutput {
    pub tokens: Vec<String>,
    pub ids: Vec<u32>,
    pub attention: Vec<u32>,
}

pub struct Embedder {
    tokenizer: Tokenizer,
    model: Session,
}

fn _get_dll_path(dll_name: &str) -> PathBuf {
    let mut exe_path = env::current_exe().expect("Failed to get exe path");
    exe_path.pop();
    exe_path.push(dll_name);
    exe_path
}

impl Embedder {
    pub fn new(tokenizer_path: &str, model_path: &str) -> anyhow::Result<Self> {
        if cfg!(target_os = "windows") {
            let dylib_path = _get_dll_path("onnxruntime.dll");
            let _ = ort::init_from(dylib_path)?.commit();
        } else if cfg!(target_os = "linux") {
            let dylib_path = _get_dll_path("libonnxruntime.so");
            let _ = ort::init_from(dylib_path)?.commit();
        }
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

        // 시스템의 사용 가능한 병렬성(코어 수)을 가져옵니다.
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let model = Session::builder()? 
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus as usize)? // 내부 쓰레드 수를 코어 수에 맞춤
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
}
