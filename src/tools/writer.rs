use std::path::PathBuf;
use std::sync::Arc;
use std::fs::File;
use arrow::array::{Float32Builder, ListBuilder, StringBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;
use crate::types::ProcessedChunk;

pub struct ParquetWriter {
    db_path: PathBuf,
    schema: Arc<Schema>,
}

impl ParquetWriter {
    pub fn new(db_path: PathBuf) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("file_id", DataType::Utf8, false),
            Field::new("chunk_id", DataType::Utf8, false),
            Field::new("index", DataType::UInt64, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("start_token", DataType::UInt64, false),
            Field::new(
                "embedding",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
        ]));
        Self { db_path, schema }
    }

    pub fn write_batch(&self, chunks: Vec<ProcessedChunk>) -> anyhow::Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let file = File::create(&self.db_path)?;
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&self.schema), None)?;

        let batch = self.chunks_to_batch(chunks)?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    pub fn create_writer(&self) -> anyhow::Result<ArrowWriter<File>> {
        let file = File::create(&self.db_path)?;
        let writer = ArrowWriter::try_new(file, Arc::clone(&self.schema), None)?;
        Ok(writer)
    }

    pub fn chunks_to_batch(&self, chunks: Vec<ProcessedChunk>) -> anyhow::Result<RecordBatch> {
        let mut file_ids = StringBuilder::new();
        let mut chunk_ids = StringBuilder::new();
        let mut indices = UInt64Builder::new();
        let mut texts = StringBuilder::new();
        let mut start_tokens = UInt64Builder::new();
        
        let mut embedding_builder = ListBuilder::new(Float32Builder::new());

        for chunk in chunks {
            file_ids.append_value(&chunk.chunk_info.file_id);
            chunk_ids.append_value(&chunk.chunk_info.chunk_id);
            indices.append_value(chunk.chunk_info.index as u64);
            texts.append_value(&chunk.chunk_info.text);
            start_tokens.append_value(chunk.chunk_info.start_token as u64);
            
            for &val in &chunk.embedding {
                embedding_builder.values().append_value(val);
            }
            embedding_builder.append(true);
        }

        let batch = RecordBatch::try_new(
            Arc::clone(&self.schema),
            vec![
                Arc::new(file_ids.finish()),
                Arc::new(chunk_ids.finish()),
                Arc::new(indices.finish()),
                Arc::new(texts.finish()),
                Arc::new(start_tokens.finish()),
                Arc::new(embedding_builder.finish()),
            ],
        )?;

        Ok(batch)
    }
}
