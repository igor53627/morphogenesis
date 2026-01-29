use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

pub const NUM_HASH_FUNCTIONS: usize = 3;
pub const STASH_SIZE: usize = 256;
pub const MAX_KICKS: usize = 500;

pub struct CuckooTable<V> {
    buckets: Vec<Option<(Vec<u8>, V)>>,
    stash: Vec<(Vec<u8>, V)>,
    num_buckets: usize,
    seeds: [u64; NUM_HASH_FUNCTIONS],
    rng_state: u64,
}

pub struct CuckooAddresser {
    num_rows: usize,
    seeds: [u64; NUM_HASH_FUNCTIONS],
}

impl CuckooAddresser {
    pub fn new(num_rows: usize) -> Self {
        Self::with_seeds(num_rows, [0x1234_5678, 0x9ABC_DEF0, 0xFEDC_BA98])
    }

    pub fn with_seeds(num_rows: usize, seeds: [u64; NUM_HASH_FUNCTIONS]) -> Self {
        Self { num_rows, seeds }
    }

    pub fn hash_indices(&self, key: &[u8]) -> [usize; NUM_HASH_FUNCTIONS] {
        let mut indices = [0usize; NUM_HASH_FUNCTIONS];
        for (i, idx) in indices.iter_mut().enumerate() {
            let mut hasher = DefaultHasher::new();
            hasher.write_u64(self.seeds[i]);
            hasher.write(key);
            *idx = (hasher.finish() as usize) % self.num_rows;
        }
        indices
    }
}

impl<V: Clone> CuckooTable<V> {
    pub fn new(num_buckets: usize) -> Self {
        Self::with_seeds(num_buckets, [0x1234_5678, 0x9ABC_DEF0, 0xFEDC_BA98])
    }

    pub fn with_seeds(num_buckets: usize, seeds: [u64; NUM_HASH_FUNCTIONS]) -> Self {
        Self {
            buckets: vec![None; num_buckets],
            stash: Vec::with_capacity(STASH_SIZE),
            num_buckets,
            seeds,
            rng_state: 0xDEAD_BEEF_CAFE_BABEu64,
        }
    }

    pub fn addresser(&self) -> CuckooAddresser {
        CuckooAddresser::with_seeds(self.num_buckets, self.seeds)
    }

    fn next_random(&mut self) -> usize {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_state >> 32) as usize
    }

    pub fn insert(&mut self, key: Vec<u8>, value: V) -> Result<usize, CuckooInsertError> {
        let indices = self.addresser().hash_indices(&key);

        for &idx in &indices {
            if self.buckets[idx].is_none() {
                self.buckets[idx] = Some((key, value));
                return Ok(idx);
            }
        }

        let mut displaced_key = key;
        let mut displaced_value = value;
        let mut last_evicted_idx: Option<usize> = None;

        for _ in 0..MAX_KICKS {
            let candidates = self.addresser().hash_indices(&displaced_key);

            let available: Vec<usize> = if let Some(last_idx) = last_evicted_idx {
                candidates
                    .iter()
                    .copied()
                    .filter(|&idx| idx != last_idx)
                    .collect()
            } else {
                candidates.to_vec()
            };

            let choice = self.next_random() % available.len();
            let idx = available[choice];

            if self.buckets[idx].is_none() {
                self.buckets[idx] = Some((displaced_key, displaced_value));
                return Ok(idx);
            }

            let existing = self.buckets[idx].take().unwrap();
            self.buckets[idx] = Some((displaced_key, displaced_value));
            displaced_key = existing.0;
            displaced_value = existing.1;
            last_evicted_idx = Some(idx);
        }

        if self.stash.len() < STASH_SIZE {
            self.stash.push((displaced_key, displaced_value));
            return Ok(usize::MAX);
        }

        Err(CuckooInsertError::TableFull)
    }

    pub fn get(&self, key: &[u8]) -> Option<(usize, &V)> {
        let indices = self.addresser().hash_indices(key);

        for &idx in &indices {
            if let Some((ref k, ref v)) = self.buckets[idx] {
                if k == key {
                    return Some((idx, v));
                }
            }
        }

        for (ref k, ref v) in &self.stash {
            if k == key {
                return Some((usize::MAX, v));
            }
        }

        None
    }

    pub fn get_actual_index(&self, key: &[u8]) -> Option<usize> {
        self.get(key).map(|(idx, _)| idx)
    }

    pub fn len(&self) -> usize {
        self.buckets.iter().filter(|b| b.is_some()).count() + self.stash.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stash_len(&self) -> usize {
        self.stash.len()
    }

    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    pub fn load_factor(&self) -> f64 {
        self.len() as f64 / self.num_buckets as f64
    }

    pub fn iter_enumerated(&self) -> impl Iterator<Item = (usize, &Vec<u8>, &V)> {
        self.buckets
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.as_ref().map(|(k, v)| (i, k, v)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CuckooInsertError {
    TableFull,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuckoo_addresser_returns_three_indices() {
        let addresser = CuckooAddresser::new(1000);
        let indices = addresser.hash_indices(b"test_key");

        assert_eq!(indices.len(), 3);
        for &idx in &indices {
            assert!(idx < 1000);
        }
    }

    #[test]
    fn cuckoo_addresser_same_key_same_indices() {
        let addresser = CuckooAddresser::new(1000);
        let indices1 = addresser.hash_indices(b"account_123");
        let indices2 = addresser.hash_indices(b"account_123");

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn cuckoo_addresser_different_keys_usually_different_indices() {
        let addresser = CuckooAddresser::new(10000);
        let indices1 = addresser.hash_indices(b"account_1");
        let indices2 = addresser.hash_indices(b"account_2");

        assert_ne!(indices1, indices2);
    }

    #[test]
    fn cuckoo_table_insert_and_get() {
        let mut table: CuckooTable<Vec<u8>> = CuckooTable::new(1000);
        let key = b"test_account".to_vec();
        let value = vec![0xAA; 32];

        let result = table.insert(key.clone(), value.clone());
        assert!(result.is_ok());

        let retrieved = table.get(&key);
        assert!(retrieved.is_some());
        let (_, v) = retrieved.unwrap();
        assert_eq!(v, &value);
    }

    #[test]
    fn cuckoo_table_get_actual_index_matches_candidate() {
        let mut table: CuckooTable<Vec<u8>> = CuckooTable::new(1000);
        let key = b"my_account".to_vec();
        let value = vec![0xBB; 32];

        table.insert(key.clone(), value).unwrap();

        let actual_idx = table.get_actual_index(&key).unwrap();
        let candidates = table.addresser().hash_indices(&key);

        assert!(
            candidates.contains(&actual_idx) || actual_idx == usize::MAX,
            "actual index {} should be one of candidates {:?} or stash",
            actual_idx,
            candidates
        );
    }

    #[test]
    fn cuckoo_table_insert_many_entries() {
        let num_entries = 5000u64;
        let table_size = num_entries as usize * 2;
        let mut table: CuckooTable<u64> = CuckooTable::new(table_size);

        for i in 0..num_entries {
            let key = format!("account_{}", i).into_bytes();
            let result = table.insert(key, i);
            assert!(result.is_ok(), "failed to insert entry {}", i);
        }

        for i in 0..num_entries {
            let key = format!("account_{}", i).into_bytes();
            let retrieved = table.get(&key);
            assert!(retrieved.is_some(), "failed to get entry {}", i);
            let (_, &v) = retrieved.unwrap();
            assert_eq!(v, i);
        }
    }

    #[test]
    fn cuckoo_stash_usage_for_high_load() {
        let num_entries = 5000u64;
        let table_size = (num_entries as f64 * 1.1) as usize;
        let mut table: CuckooTable<u64> = CuckooTable::new(table_size);
        let mut stash_count = 0;

        for i in 0..num_entries {
            let key = format!("account_{}", i).into_bytes();
            if let Ok(idx) = table.insert(key, i) {
                if idx == usize::MAX {
                    stash_count += 1;
                }
            }
        }

        assert!(
            stash_count <= STASH_SIZE,
            "stash overflow: {} > {}",
            stash_count,
            STASH_SIZE
        );
    }

    #[test]
    fn cuckoo_achieves_85_percent_load_factor() {
        let num_entries = 85_000u64;
        let table_size = 100_000;
        let mut table: CuckooTable<u64> = CuckooTable::new(table_size);
        let mut inserted = 0u64;
        let mut stash_count = 0usize;

        for i in 0..num_entries {
            let key = format!("account_{}", i).into_bytes();
            match table.insert(key, i) {
                Ok(idx) => {
                    inserted += 1;
                    if idx == usize::MAX {
                        stash_count += 1;
                    }
                }
                Err(_) => break,
            }
        }

        let load_factor = inserted as f64 / table_size as f64;
        assert!(
            load_factor >= 0.85,
            "load factor {} should be >= 0.85 (inserted {}/{})",
            load_factor,
            inserted,
            table_size
        );
        assert!(
            stash_count <= STASH_SIZE,
            "stash {} should be <= {}",
            stash_count,
            STASH_SIZE
        );
    }

    #[test]
    fn cuckoo_achieves_90_percent_load_factor() {
        let num_entries = 90_000u64;
        let table_size = 100_000;
        let mut table: CuckooTable<u64> = CuckooTable::new(table_size);
        let mut inserted = 0u64;

        for i in 0..num_entries {
            let key = format!("account_{}", i).into_bytes();
            match table.insert(key, i) {
                Ok(_) => inserted += 1,
                Err(_) => break,
            }
        }

        let load_factor = inserted as f64 / table_size as f64;
        assert!(
            load_factor >= 0.90,
            "load factor {} should be >= 0.90 (inserted {}/{})",
            load_factor,
            inserted,
            table_size
        );
    }
}
