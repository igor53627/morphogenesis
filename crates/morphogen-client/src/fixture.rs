//! PIR2 fixture reader for e2e tests.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

pub const STATE_MAGIC: [u8; 4] = *b"PIR2";
pub const STATE_HEADER_SIZE: usize = 64;
pub const STATE_ENTRY_SIZE: usize = 84;
pub const STEM_INDEX_ENTRY_SIZE: usize = 39;

#[derive(Debug, Clone)]
pub struct StateHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub entry_size: u16,
    pub entry_count: u64,
    pub block_number: u64,
    pub chain_id: u64,
    pub block_hash: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct StateEntry {
    pub address: [u8; 20],
    pub tree_index: [u8; 32],
    pub value: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct StemIndexEntry {
    pub stem: [u8; 31],
    pub offset: u64,
}

pub struct FixtureReader {
    header: StateHeader,
    state_file: BufReader<File>,
    stems: Vec<StemIndexEntry>,
}

impl FixtureReader {
    pub fn open(state_path: &Path, stem_index_path: &Path) -> std::io::Result<Self> {
        let mut state_file = BufReader::new(File::open(state_path)?);

        let header = Self::read_header(&mut state_file)?;
        if header.magic != STATE_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid PIR2 magic",
            ));
        }

        let stems = Self::read_stem_index(stem_index_path)?;

        Ok(Self {
            header,
            state_file,
            stems,
        })
    }

    fn read_header(reader: &mut BufReader<File>) -> std::io::Result<StateHeader> {
        let mut buf = [0u8; STATE_HEADER_SIZE];
        reader.read_exact(&mut buf)?;

        Ok(StateHeader {
            magic: buf[0..4].try_into().unwrap(),
            version: u16::from_le_bytes(buf[4..6].try_into().unwrap()),
            entry_size: u16::from_le_bytes(buf[6..8].try_into().unwrap()),
            entry_count: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            block_number: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            chain_id: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
            block_hash: buf[32..64].try_into().unwrap(),
        })
    }

    fn read_stem_index(path: &Path) -> std::io::Result<Vec<StemIndexEntry>> {
        let mut file = BufReader::new(File::open(path)?);
        let mut count_buf = [0u8; 8];
        file.read_exact(&mut count_buf)?;
        let count = u64::from_le_bytes(count_buf) as usize;

        let mut stems = Vec::with_capacity(count);
        for _ in 0..count {
            let mut entry_buf = [0u8; STEM_INDEX_ENTRY_SIZE];
            file.read_exact(&mut entry_buf)?;

            stems.push(StemIndexEntry {
                stem: entry_buf[0..31].try_into().unwrap(),
                offset: u64::from_le_bytes(entry_buf[31..39].try_into().unwrap()),
            });
        }

        Ok(stems)
    }

    pub fn header(&self) -> &StateHeader {
        &self.header
    }

    pub fn entry_count(&self) -> u64 {
        self.header.entry_count
    }

    pub fn stem_count(&self) -> usize {
        self.stems.len()
    }

    pub fn read_entry(&mut self, index: u64) -> std::io::Result<StateEntry> {
        let offset = STATE_HEADER_SIZE as u64 + index * STATE_ENTRY_SIZE as u64;
        self.state_file.seek(SeekFrom::Start(offset))?;

        let mut buf = [0u8; STATE_ENTRY_SIZE];
        self.state_file.read_exact(&mut buf)?;

        Ok(StateEntry {
            address: buf[0..20].try_into().unwrap(),
            tree_index: buf[20..52].try_into().unwrap(),
            value: buf[52..84].try_into().unwrap(),
        })
    }

    pub fn read_entries(&mut self, start: u64, count: usize) -> std::io::Result<Vec<StateEntry>> {
        let mut entries = Vec::with_capacity(count);
        for i in 0..count as u64 {
            if start + i >= self.header.entry_count {
                break;
            }
            entries.push(self.read_entry(start + i)?);
        }
        Ok(entries)
    }

    pub fn iter_entries(&mut self) -> impl Iterator<Item = std::io::Result<StateEntry>> + '_ {
        (0..self.header.entry_count).map(move |i| self.read_entry(i))
    }

    pub fn stems(&self) -> &[StemIndexEntry] {
        &self.stems
    }

    pub fn entries_for_stem(&mut self, stem_idx: usize) -> std::io::Result<Vec<StateEntry>> {
        if stem_idx >= self.stems.len() {
            return Ok(vec![]);
        }

        let start = self.stems[stem_idx].offset;
        let end = if stem_idx + 1 < self.stems.len() {
            self.stems[stem_idx + 1].offset
        } else {
            self.header.entry_count
        };

        self.read_entries(start, (end - start) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fixtures")
    }

    #[test]
    fn read_fixture_header() {
        let state_path = fixture_path().join("state.bin");
        let stem_path = fixture_path().join("stem-index.bin");

        if !state_path.exists() {
            eprintln!("Skipping test: fixture not found at {:?}", state_path);
            return;
        }

        let reader = FixtureReader::open(&state_path, &stem_path).unwrap();
        let header = reader.header();

        assert_eq!(header.magic, *b"PIR2");
        assert_eq!(header.version, 1);
        assert_eq!(header.entry_size, 84);
        assert_eq!(header.entry_count, 102222);
        assert_eq!(header.block_number, 10021930);
        assert_eq!(header.chain_id, 11155111);
    }

    #[test]
    fn read_fixture_stems() {
        let state_path = fixture_path().join("state.bin");
        let stem_path = fixture_path().join("stem-index.bin");

        if !state_path.exists() {
            return;
        }

        let reader = FixtureReader::open(&state_path, &stem_path).unwrap();
        assert_eq!(reader.stem_count(), 11282);
    }

    #[test]
    fn read_fixture_entries() {
        let state_path = fixture_path().join("state.bin");
        let stem_path = fixture_path().join("stem-index.bin");

        if !state_path.exists() {
            return;
        }

        let mut reader = FixtureReader::open(&state_path, &stem_path).unwrap();
        let entries = reader.read_entries(0, 10).unwrap();

        assert_eq!(entries.len(), 10);
        for entry in &entries {
            assert!(
                entry.address.iter().any(|&b| b != 0) || entry.tree_index.iter().any(|&b| b != 0)
            );
        }
    }

    #[test]
    fn read_entries_for_stem() {
        let state_path = fixture_path().join("state.bin");
        let stem_path = fixture_path().join("stem-index.bin");

        if !state_path.exists() {
            return;
        }

        let mut reader = FixtureReader::open(&state_path, &stem_path).unwrap();
        let entries = reader.entries_for_stem(0).unwrap();

        assert!(!entries.is_empty());
    }
}
