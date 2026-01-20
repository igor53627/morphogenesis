fn main() {
    println!("cargo:rerun-if-changed=cuda/fused_kernel.cu");

    // Only compile CUDA if nvcc is available AND cuda feature is enabled
    #[cfg(feature = "cuda")]
    if std::process::Command::new("nvcc").arg("--version").output().is_ok() {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_75".to_string());
        
        println!("cargo:warning=Compiling for CUDA architecture: {}", arch);

        let status = std::process::Command::new("nvcc")
            .arg("-ptx")
            .arg("-O3")
            .arg(format!("-arch={}", arch))
            .arg("cuda/fused_kernel.cu")
            .arg("-o")
            .arg(format!("{}/fused_kernel.ptx", out_dir))
            .status()
            .unwrap();

        if !status.success() {
            panic!("Failed to compile CUDA kernel to PTX");
        }
            
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=dylib=cudart");
    } else {
        println!("cargo:warning=nvcc not found, skipping CUDA kernel compilation");
    }
}
