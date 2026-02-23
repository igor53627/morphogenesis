//! GPU-resident page storage for PIR.
//!
//! This module provides the `GpuPageMatrix` which manages database pages
//! in GPU device memory.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice};
#[cfg(feature = "cuda")]
use morphogen_storage::ChunkedMatrix;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Page size in bytes (4KB).
pub const PAGE_SIZE_BYTES: usize = 4096;

/// A matrix of pages resident in GPU memory.
#[cfg(feature = "cuda")]
pub struct GpuPageMatrix {
    device: Arc<CudaDevice>,
    data: CudaSlice<u8>,
    num_pages: usize,
}

#[cfg(feature = "cuda")]
impl GpuPageMatrix {
    /// Create a new GpuPageMatrix from a CPU buffer.
    pub fn new(
        device: Arc<CudaDevice>,
        cpu_data: &[u8],
    ) -> Result<Self, cudarc::driver::DriverError> {
        if cpu_data.len() % PAGE_SIZE_BYTES != 0 {
            panic!("cpu_data length must be multiple of PAGE_SIZE_BYTES");
        }

        let num_pages = cpu_data.len() / PAGE_SIZE_BYTES;
        let data = device.htod_sync_copy(cpu_data)?;

        Ok(Self {
            device,
            data,
            num_pages,
        })
    }

    /// Create a new GpuPageMatrix from a ChunkedMatrix (CPU).
    pub fn from_chunked_matrix(
        device: Arc<CudaDevice>,
        matrix: &ChunkedMatrix,
    ) -> Result<Self, cudarc::driver::DriverError> {
        let total_size = matrix.total_size_bytes();
        // Pad total size if not aligned
        let padding = if total_size % PAGE_SIZE_BYTES != 0 {
            PAGE_SIZE_BYTES - (total_size % PAGE_SIZE_BYTES)
        } else {
            0
        };
        let padded_total_size = total_size + padding;
        let num_pages = padded_total_size / PAGE_SIZE_BYTES;

        let mut gpu_matrix = Self::alloc_empty(device.clone(), num_pages)?;

        let mut current_offset = 0;
        for i in 0..matrix.num_chunks() {
            let chunk = matrix.chunk(i);
            let size = matrix.chunk_size(i);

            // Check if chunk size is multiple of PAGE_SIZE_BYTES
            // If it's the last chunk, it might not be multiple.
            if size % PAGE_SIZE_BYTES != 0 {
                // If not multiple, we need to pad this chunk before upload
                let mut padded_chunk = chunk.as_slice().to_vec();
                let chunk_padding = PAGE_SIZE_BYTES - (size % PAGE_SIZE_BYTES);
                padded_chunk.extend(std::iter::repeat(0).take(chunk_padding));

                let start_page = current_offset / PAGE_SIZE_BYTES;
                gpu_matrix.update_pages(start_page, &padded_chunk)?;
            } else {
                let start_page = current_offset / PAGE_SIZE_BYTES;
                gpu_matrix.update_pages(start_page, chunk.as_slice())?;
            }

            current_offset += size;
        }

        Ok(gpu_matrix)
    }

    /// Allocate an empty (zeroed) GpuPageMatrix on the device.
    pub fn alloc_empty(
        device: Arc<CudaDevice>,
        num_pages: usize,
    ) -> Result<Self, cudarc::driver::DriverError> {
        let size_bytes = num_pages * PAGE_SIZE_BYTES;
        let data = device.alloc_zeros::<u8>(size_bytes)?;

        Ok(Self {
            device,
            data,
            num_pages,
        })
    }

    /// Update a range of pages from the CPU.
    pub fn update_pages(
        &mut self,
        start_page: usize,
        cpu_pages: &[u8],
    ) -> Result<(), cudarc::driver::DriverError> {
        if cpu_pages.len() % PAGE_SIZE_BYTES != 0 {
            panic!("cpu_pages length must be multiple of PAGE_SIZE_BYTES");
        }

        let num_update_pages = cpu_pages.len() / PAGE_SIZE_BYTES;
        if start_page + num_update_pages > self.num_pages {
            panic!("Update range exceeds matrix bounds");
        }

        let offset = start_page * PAGE_SIZE_BYTES;
        // Get a sub-slice for the update range
        let mut sub_slice = self.data.slice_mut(offset..offset + cpu_pages.len());
        self.device.htod_sync_copy_into(cpu_pages, &mut sub_slice)?;

        Ok(())
    }

    /// Get the number of pages in the matrix.
    pub fn num_pages(&self) -> usize {
        self.num_pages
    }

    /// Get the total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.num_pages * PAGE_SIZE_BYTES
    }

    /// Access the underlying CUDA slice.
    pub fn as_slice(&self) -> &CudaSlice<u8> {
        &self.data
    }

    /// Access the underlying CUDA device for matrix replacement workflows.
    pub fn device(&self) -> Arc<CudaDevice> {
        Arc::clone(&self.device)
    }
}

#[cfg(not(feature = "cuda"))]
pub struct GpuPageMatrix {
    _num_pages: usize,
}

#[cfg(not(feature = "cuda"))]
impl GpuPageMatrix {
    pub fn new(_cpu_data: &[u8]) -> Result<Self, &'static str> {
        Err("CUDA feature not enabled")
    }

    pub fn num_pages(&self) -> usize {
        0
    }
}
