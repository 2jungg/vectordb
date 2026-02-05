use std::{env, path::PathBuf, thread};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use MyVectorDB::tools::embedder::Embedder;
use MyVectorDB::tools::text_splitter::TextSplitter;
use MyVectorDB::tools::writer::ParquetWriter;
use MyVectorDB::types::{VectorChunk, FileMap, ProcessedChunk};
use uuid::Uuid;

fn get_data_dir() -> PathBuf {
    let args: Vec<String> = env::args().collect();
    let current_dir = env::current_dir().expect("Failed to get current directory");
    
    if args.len() > 1 {
        let path = PathBuf::from(&args[1]);
        if path.is_absolute() {
            path
        } else {
            current_dir.join(path)
        }
    } else {
        current_dir.join("data")
    }
}

fn get_file_list(dir: &PathBuf) -> Vec<FileMap> {
    let mut file_list = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir).expect("Failed to read directory") {
            let entry = entry.expect("Failed to get directory entry");
            let path = entry.path();
            if path.is_file() && _is_supported_file(path.to_str().unwrap_or("")) {
                let file_name = path.to_str().unwrap().to_string();
                let file_id = Uuid::new_v4().to_string();
                file_list.push(FileMap::new(file_name, file_id));
            }
        }
    }
    file_list
}

fn _is_supported_file(file_name: &str) -> bool {
    let supported_extensions = ["txt", "md"];
    if let Some(ext) = file_name.split('.').last() {
        supported_extensions.contains(&ext)
    } else {
        false
    }
}

fn make_parquet_db(
    db_path: PathBuf,
    file_list: Vec<FileMap>,
    text_splitter: Arc<TextSplitter>,
    mut embedder: Embedder,
) {
    println!("Creating Parquet DB at: {:?}", db_path);
    
    let (tx_chunk, rx_chunk): (Sender<VectorChunk>, Receiver<VectorChunk>) = channel();
    let (tx_processed, rx_processed): (Sender<ProcessedChunk>, Receiver<ProcessedChunk>) = channel();

    // 1. Splitter Thread
    let splitter_handle = thread::spawn(move || {
        for file in file_list {
            println!("Splitting file: {:?}", file.file_name);
            text_splitter.split(&file, &tx_chunk).expect("Failed to split text");
        }
        drop(tx_chunk); // Signal end of splitting
    });

    // 2. Embedder Thread
    let embedder_handle = thread::spawn(move || {
        while let Ok(chunk) = rx_chunk.recv() {
            let embedding = embedder.get_embedding(&chunk.text).expect("Failed to get embedding");
            let processed = ProcessedChunk::new(chunk, embedding);
            tx_processed.send(processed).expect("Failed to send processed chunk");
        }
        drop(tx_processed);
    });

    // 3. Writer (Current Thread)
    let writer = ParquetWriter::new(db_path);
    let mut all_chunks = Vec::new();
    
    while let Ok(processed) = rx_processed.recv() {
        all_chunks.push(processed);
    }
    
    writer.write_batch(all_chunks).expect("Failed to write to Parquet");

    splitter_handle.join().expect("Splitter thread panicked");
    embedder_handle.join().expect("Embedder thread panicked");
    
    println!("DB creation completed.");
}

fn main() {
    let data_dir = get_data_dir();
    println!("Data directory is set to: {:?}", data_dir);
    
    let file_list = get_file_list(&data_dir);
    println!("{} files found in the data directory.", file_list.len());
    
    let text_splitter = Arc::new(
        TextSplitter::new("bge-m3/onnx/tokenizer.json", 512, 50)
            .expect("Failed to create TextSplitter")
    );
    
    let embedder = Embedder::new(
        "bge-m3/onnx/tokenizer.json", 
        "bge-m3/onnx/model.onnx"
    ).expect("Failed to create Embedder");

    let db_path = data_dir.join("output.parquet");
    
    make_parquet_db(db_path, file_list, text_splitter, embedder);
}
