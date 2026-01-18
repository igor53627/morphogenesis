pub const ROW_SIZE_PRIVACY_ONLY: usize = 256;
pub const ROW_SIZE_TRUSTLESS: usize = 2048;

pub const ROW_SIZE_BYTES: usize = ROW_SIZE_PRIVACY_ONLY;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueryMode {
    #[default]
    PrivacyOnly,
    Trustless,
}

impl QueryMode {
    pub const fn row_size(&self) -> usize {
        match self {
            QueryMode::PrivacyOnly => ROW_SIZE_PRIVACY_ONLY,
            QueryMode::Trustless => ROW_SIZE_TRUSTLESS,
        }
    }
}
