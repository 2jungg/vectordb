struct VectorChunk {
    file_id: String,
    chunk_id: String,
    index: usize,
    text: String,
    start_token: usize,
}

struct FileMap {
    file_name: String,
    file_id: String,
}

struct EmbeddedData {
    chunk_id: String,
    embedding: Vec<f32>,
}