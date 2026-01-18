use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;
use std::slice;

pub struct AlignedMatrix {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

impl AlignedMatrix {
    pub fn new(size_bytes: usize) -> Self {
        let layout = Layout::from_size_align(size_bytes.max(1), 64).expect("invalid layout");

        let ptr = if size_bytes == 0 {
            NonNull::dangling()
        } else {
            let raw = unsafe { alloc_zeroed(layout) };
            NonNull::new(raw).expect("allocation failed")
        };

        Self {
            ptr,
            len: size_bytes,
            layout,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    pub fn copy_from(src: &AlignedMatrix) -> Self {
        let mut dst = Self::new(src.len);
        if src.len != 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), src.len);
            }
        }
        dst
    }
}

impl Drop for AlignedMatrix {
    fn drop(&mut self) {
        if self.len != 0 {
            unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
        }
    }
}

// SAFETY: AlignedMatrix owns its memory exclusively and only provides
// immutable access via as_ptr/as_slice when shared. The underlying
// memory is valid for the lifetime of the struct.
unsafe impl Send for AlignedMatrix {}
unsafe impl Sync for AlignedMatrix {}

#[cfg(test)]
mod tests {
    use super::AlignedMatrix;

    #[test]
    fn aligned_matrix_is_64b_aligned() {
        let matrix = AlignedMatrix::new(128);
        let ptr = matrix.as_ptr() as usize;
        assert_eq!(ptr % 64, 0);
    }
}
