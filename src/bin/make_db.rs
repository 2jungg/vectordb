use std::{collections::HashMap, env, fs, path::PathBuf, thread};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde_json;

use MyVectorDB::tools::embedder::Embedder;
use MyVectorDB::tools::text_splitter::TextSplitter;
use MyVectorDB::tools::writer::ParquetWriter;
use MyVectorDB::types::{TextChunk, FileMap, EmbeddedChunk};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

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
            if path.is_file() && is_supported_file(path.to_str().unwrap_or("")) {
                let file_name = path.to_str().unwrap().to_string();
                
                let mut hasher = DefaultHasher::new();
                file_name.hash(&mut hasher);
                let file_id = format!("{:x}", hasher.finish());
                
                file_list.push(FileMap::new(file_name, file_id));
            }
        }
    }
    file_list
}

fn is_supported_file(file_name: &str) -> bool {
    let supported_extensions = ["txt", "md"];
    if let Some(ext) = file_name.split(".").last() {
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
    let num_files = file_list.len();
    
    let (tx_chunk, rx_chunk): (Sender<TextChunk>, Receiver<TextChunk>) = channel();
    let (tx_processed, rx_processed): (Sender<EmbeddedChunk>, Receiver<EmbeddedChunk>) = channel();

    let multi = MultiProgress::new();
    let style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
        .expect("Failed to set progress bar style")
        .progress_chars("#>-");

    let pb_splitter = multi.add(ProgressBar::new(num_files as u64));
    pb_splitter.set_style(style.clone());
    pb_splitter.set_message("Files Split");

    let pb_embedder = multi.add(ProgressBar::new_spinner());
    pb_embedder.set_style(ProgressStyle::default_spinner()
        .template("{spinner:.yellow} [{elapsed_precise}] Embedding: {pos} chunks {msg}")
        .expect("Failed to set spinner style"));

    let pb_writer = multi.add(ProgressBar::new_spinner());
    pb_writer.set_style(ProgressStyle::default_spinner()
        .template("{spinner:.magenta} [{elapsed_precise}] Writing: {pos} chunks {msg}")
        .expect("Failed to set spinner style"));

    let splitter_handle = thread::spawn(move || {
        for file in file_list {
            text_splitter.split(&file, &tx_chunk).expect("Failed to split text");
            pb_splitter.inc(1);
        }
        pb_splitter.finish_with_message("Splitting completed");
        drop(tx_chunk); 
    });

    let pb_embedder_c = pb_embedder.clone();
    let embedder_handle = thread::spawn(move || {
        while let Ok(chunk) = rx_chunk.recv() {
            let embedding = embedder.get_embedding(&chunk.text).expect("Failed to get embedding");
            let processed = EmbeddedChunk::new(chunk, embedding);
            tx_processed.send(processed).expect("Failed to send processed chunk");
            pb_embedder_c.inc(1);
        }
        pb_embedder_c.finish_with_message("Embedding completed");
        drop(tx_processed);
    });

    let writer_tool = ParquetWriter::new(db_path);
    let mut arrow_writer = writer_tool.create_writer().expect("Failed to create arrow writer");
    
    let mut current_batch = Vec::new();
    const BATCH_SIZE: usize = 100;
    let mut total_processed = 0;

    while let Ok(processed) = rx_processed.recv() {
        current_batch.push(processed);
        total_processed += 1;
        pb_writer.inc(1);

        if current_batch.len() >= BATCH_SIZE {
            let batch = writer_tool.chunks_to_batch(current_batch).expect("Failed to convert chunks to batch");
            arrow_writer.write(&batch).expect("Failed to write batch to Parquet");
            arrow_writer.flush().expect("Failed to flush parquet writer");
            current_batch = Vec::new();
            pb_writer.set_message(format!("(Batch saved, total: {})", total_processed));
        }
    }
    
    if !current_batch.is_empty() {
        let batch = writer_tool.chunks_to_batch(current_batch).expect("Failed to convert chunks to batch");
        arrow_writer.write(&batch).expect("Failed to write final batch to Parquet");
    }
    
    arrow_writer.close().expect("Failed to close arrow writer");
    pb_writer.finish_with_message(format!("Writing completed. Total: {}", total_processed));

    splitter_handle.join().expect("Splitter thread panicked");
    embedder_handle.join().expect("Embedder thread panicked");
}

fn main() {
    let data_dir = get_data_dir();
    println!("Data directory is set to: {:?}", data_dir);
    
    let file_list = get_file_list(&data_dir);
    println!("{} files found in the data directory.", file_list.len());

    let mut file_map_data = HashMap::new();
    for file_map_item in &file_list {
        file_map_data.insert(file_map_item.file_id.clone(), file_map_item.file_name.clone());
    }

    let file_map_path = data_dir.join("file_map.json");
    let json_string = serde_json::to_string_pretty(&file_map_data)
        .expect("Failed to serialize file map to JSON");
    fs::write(&file_map_path, json_string)
        .expect("Failed to write file_map.json");
    println!("File map saved to: {:?}", file_map_path);
    
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