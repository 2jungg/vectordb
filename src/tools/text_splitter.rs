use tokenizers::Tokenizer;
use memmap2::Mmap;
use std::fs::File;
use std::sync::mpsc::Sender;
use crate::types::{ VectorChunk, FileMap };
use uuid::Uuid;
use encoding_rs;

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
    
    pub fn split(&self, file_map: &FileMap, tx: &Sender<VectorChunk>) -> anyhow::Result<()> {
        let file = File::open(&file_map.file_name)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let (cow_text, _, _) = encoding_rs::EUC_KR.decode(&mmap);
        let text = cow_text.to_string();

        self._process_chunks(file_map, &text, tx)?;

        Ok(())
    }

    fn _process_chunks(&self, file_map: &FileMap, full_text: &str, tx: &Sender<VectorChunk>) -> anyhow::Result<()> {
        let encoding = self.tokenizer.encode(full_text, true).expect("Encoding failed");
        let ids = encoding.get_ids();

        let mut start = 0;
        let mut idx = 0;
        while start < ids.len() {
            let end = (start + self.max_tokens).min(ids.len());
            let chunk_ids = &ids[start..end];
            
            let chunk_text = self.tokenizer.decode(chunk_ids, true).unwrap();
            let vector_chunk = VectorChunk::new(
                file_map.file_id.clone(),
                format!("chunk_{}", Uuid::new_v4()),
                idx,
                chunk_text,
                start
            );
            tx.send(vector_chunk).expect("Failed to send VectorChunk");
            idx += 1;

            if end == ids.len() { break; }
            start += self.max_tokens - self.overlap_tokens;
        }

        Ok(())
    }
}