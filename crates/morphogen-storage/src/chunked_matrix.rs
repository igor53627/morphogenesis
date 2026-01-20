use std::sync::Arc;

use crate::AlignedMatrix;

pub struct ChunkedMatrix {
    chunks: Vec<Arc<AlignedMatrix>>,
    chunk_sizes: Vec<usize>,
    chunk_size_bytes: usize,
    total_size_bytes: usize,
}

impl ChunkedMatrix {
    pub fn new(total_size_bytes: usize, chunk_size_bytes: usize) -> Self {
        assert!(chunk_size_bytes > 0, "chunk size must be non-zero");

        let mut chunks = Vec::new();
        let mut chunk_sizes = Vec::new();
        let mut remaining = total_size_bytes;

        while remaining > 0 {
            let size = remaining.min(chunk_size_bytes);
            chunks.push(Arc::new(AlignedMatrix::new(size)));
            chunk_sizes.push(size);
            remaining -= size;
        }

        if total_size_bytes == 0 {
            chunks.push(Arc::new(AlignedMatrix::new(0)));
            chunk_sizes.push(0);
        }

        Self {
            chunks,
            chunk_sizes,
            chunk_size_bytes,
            total_size_bytes,
        }
    }

    pub fn from_chunks(
        chunks: Vec<Arc<AlignedMatrix>>,
        chunk_sizes: Vec<usize>,
        chunk_size_bytes: usize,
        total_size_bytes: usize,
    ) -> Self {
        assert_eq!(chunks.len(), chunk_sizes.len(), "chunk metadata mismatch");
        Self {
            chunks,
            chunk_sizes,
            chunk_size_bytes,
            total_size_bytes,
        }
    }

    pub fn total_size_bytes(&self) -> usize {
        self.total_size_bytes
    }

    pub fn chunk_size_bytes(&self) -> usize {
        self.chunk_size_bytes
    }

    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    pub fn chunk(&self, index: usize) -> &Arc<AlignedMatrix> {
        &self.chunks[index]
    }

    pub fn chunk_size(&self, index: usize) -> usize {
        self.chunk_sizes[index]
    }

    pub fn chunks(&self) -> &[Arc<AlignedMatrix>] {
        &self.chunks
    }

    pub fn chunk_sizes(&self) -> &[usize] {
        &self.chunk_sizes
    }

    pub fn fill_with_pattern(&mut self, seed: u64) {
        let mut state = seed;
        for chunk in &mut self.chunks {
            let chunk = Arc::get_mut(chunk).expect("matrix must be uniquely owned");
            let bytes = chunk.as_mut_slice();
            for byte in bytes.iter_mut() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *byte = (state >> 56) as u8;
            }
        }
    }

    pub fn write_row(&mut self, row_idx: usize, row_size: usize, data: &[u8]) {
        let row_offset = row_idx * row_size;
        let chunk_idx = row_offset / self.chunk_size_bytes;
        let offset_in_chunk = row_offset % self.chunk_size_bytes;

        let chunk =
            Arc::get_mut(&mut self.chunks[chunk_idx]).expect("chunk must be uniquely owned");
        let slice = chunk.as_mut_slice();
        let len = data.len().min(row_size);
        slice[offset_in_chunk..offset_in_chunk + len].copy_from_slice(&data[..len]);
    }

    pub fn write_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        for chunk in &self.chunks {
            file.write_all(chunk.as_slice())?;
        }
        Ok(())
    }
}
