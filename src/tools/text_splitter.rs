use tokenizers::Tokenizer;
use std::io;
use memmap2::Mmap;
use std::{ fs::File, path::PathBuf };
use std::sync::mpsc::Sender;

pub struct TextSplitter {
    tokenizer: Tokenizer,
    max_tokens: usize,
    overlap_tokens : usize
}

impl TextSplitter {
    pub fn new(tokenizer_path: &str, max_tokens: usize, overlap_tokens: usize) -> anyhow::Result<Self>{
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;
        let max_tokens = max_tokens;
        let overlap_tokens = overlap_tokens;
        Ok(Self{ tokenizer, max_tokens, overlap_tokens })
    }
    
    pub fn split(&self, file_path: &PathBuf) -> anyhow::Result<()> {
        let file = File::open(file_path);
        let mmap = unsafe { Mmap::map(&file?)? };

        let text = std::str::from_utf8(&mmap)?;

        self._process_chunks(text); 

        Ok(())
    }

    fn _process_chunks(&self, full_text: &str) {
        let encoding = self.tokenizer.encode(full_text, true).expect("Encoding failed");
        let ids = encoding.get_ids();

        let mut start = 0;
        while start < ids.len() {
            let end = (start + self.max_tokens).min(ids.len());
            let chunk_ids = &ids[start..end];
            
            let chunk_text = self.tokenizer.decode(chunk_ids, true).unwrap();
            self.send_to_vector_db(&chunk_text);

            if end == ids.len() { break; }
            start += self.max_tokens - self.overlap_tokens;
        }
    }

    fn send_to_vector_db(&self, chunk: &str) {
        let preview = chunk.chars().take(100).collect::<String>();
        println!("Sending chunk to Vector DB: {}", preview);
    }
}