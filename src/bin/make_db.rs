use std::{env, path::PathBuf};
use MyVectorDB::tools::convert::TextProcessor;
use MyVectorDB::tools::text_splitter::TextSplitter;
use std::sync::mpsc::{channel, Receiver, Sender};

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

fn get_file_list(dir: &PathBuf) -> Vec<PathBuf> {
    let mut file_list = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir).expect("Failed to read directory") {
            let entry = entry.expect("Failed to get directory entry");
            let path = entry.path();
            if path.is_file() && _is_supported_file(path.to_str().unwrap_or("")) {
                file_list.push(path);
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

fn make_parquet_db(db_path: &PathBuf, file_list: &Vec<PathBuf>, text_splitter: &TextSplitter) {
    println!("Creating Parquet DB at: {:?}", db_path);
    for file in file_list {
        println!("Processing file: {:?}", file);
        text_splitter.split(file);
    }
}

fn main() {
    let data_dir = get_data_dir();
    println!("Data directory is set to: {:?}", data_dir);
    let file_list = get_file_list(&data_dir);
    println!("{:} files found in the data directory.", file_list.len());
    let text_splitter = TextSplitter::new("bge-m3/onnx/tokenizer.json", 2048, 205).expect("Failed to create TextSplitter");
    let db_path = data_dir.join("output.parquet");
    make_parquet_db(&db_path, &file_list, &text_splitter);

}