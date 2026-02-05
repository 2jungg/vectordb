#[derive(Clone)]
pub struct VectorChunk {
    pub file_id: String,
    pub chunk_id: String,
    pub index: usize,
    pub text: String,
    pub start_token: usize,
}

impl VectorChunk {
    pub fn new(file_id: String, chunk_id: String, index: usize, text: String, start_token: usize) -> Self {
        Self {
            file_id,
            chunk_id,
            index,
            text,
            start_token,
        }
    }
}

#[derive(Clone)]
pub struct FileMap {
    pub file_name: String,
    pub file_id: String,
}

impl FileMap {
    pub fn new(file_name: String, file_id: String) -> Self {
        Self { file_name, file_id }
    }
}

pub struct ProcessedChunk {
    pub chunk_info: VectorChunk,
    pub embedding: Vec<f32>,
}

impl ProcessedChunk {
    pub fn new(chunk_info: VectorChunk, embedding: Vec<f32>) -> Self {
        Self {
            chunk_info,
            embedding,
        }
    }
}