#[cfg(feature = "profiling")]
use std::time::Instant;

#[cfg(feature = "profiling")]
pub struct Profiler {
    timings: Vec<(String, f64)>,
    start: Instant,
}

#[cfg(feature = "profiling")]
impl Profiler {
    pub fn new() -> Self {
        Self {
            timings: Vec::new(),
            start: Instant::now(),
        }
    }

    pub fn checkpoint(&mut self, label: &str) {
        let elapsed = self.start.elapsed().as_secs_f64() * 1000.0; // ms
        self.timings.push((label.to_string(), elapsed));
        self.start = Instant::now();
    }

    pub fn report(&self) -> String {
        let mut report = String::from("Profiling breakdown:\n");
        for (label, ms) in &self.timings {
            report.push_str(&format!("  {}: {:.2}ms\n", label, ms));
        }
        report
    }

    #[allow(dead_code)]
    pub fn get_timings(&self) -> &[(String, f64)] {
        &self.timings
    }
}

#[cfg(not(feature = "profiling"))]
pub struct Profiler;

#[cfg(not(feature = "profiling"))]
impl Profiler {
    pub fn new() -> Self {
        Self
    }

    pub fn checkpoint(&mut self, _label: &str) {
        // No-op when profiling is disabled
    }

    pub fn report(&self) -> String {
        String::new()
    }

    pub fn get_timings(&self) -> &[(String, f64)] {
        &[]
    }
}
