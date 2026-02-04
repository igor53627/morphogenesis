fn main() {
    println!("cargo:rerun-if-changed=cuda/fused_kernel.cu");
    println!("cargo:rerun-if-changed=cuda/fused_kernel_optimized.cu");

    // Only compile CUDA if nvcc is available AND cuda feature is enabled
    #[cfg(feature = "cuda")]
    if std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok()
    {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_75".to_string());

        println!("cargo:warning=Compiling for CUDA architecture: {}", arch);

        // Compile original kernel
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

        // Compile optimized kernel v1 (best effort)
        let opt_status = std::process::Command::new("nvcc")
            .arg("-ptx")
            .arg("-O3")
            .arg(format!("-arch={}", arch))
            .arg("cuda/fused_kernel_optimized.cu")
            .arg("-o")
            .arg(format!("{}/fused_kernel_optimized.ptx", out_dir))
            .status();

        if let Ok(st) = opt_status {
            if st.success() {
                println!("cargo:warning=Compiled optimized CUDA kernel v1");
            }
        }

        // Compile optimized kernel v2 - minimal shared memory (best effort)
        let opt_v2_status = std::process::Command::new("nvcc")
            .arg("-ptx")
            .arg("-O3")
            .arg(format!("-arch={}", arch))
            .arg("cuda/fused_kernel_optimized_v2.cu")
            .arg("-o")
            .arg(format!("{}/fused_kernel_optimized_v2.ptx", out_dir))
            .status();

        if let Ok(st) = opt_v2_status {
            if st.success() {
                println!("cargo:warning=Compiled optimized CUDA kernel v2 (minimal shared memory)");
            }
        }

        // Compile optimized kernel v3 - hybrid approach (best effort)
        let opt_v3_status = std::process::Command::new("nvcc")
            .arg("-ptx")
            .arg("-O3")
            .arg(format!("-arch={}", arch))
            .arg("cuda/fused_kernel_optimized_v3.cu")
            .arg("-o")
            .arg(format!("{}/fused_kernel_optimized_v3.ptx", out_dir))
            .status();

        if let Ok(st) = opt_v3_status {
            if st.success() {
                println!("cargo:warning=Compiled optimized CUDA kernel v3 (hybrid approach)");
            }
        }

        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=dylib=cudart");
    } else {
        println!("cargo:warning=nvcc not found, skipping CUDA kernel compilation");
    }
}
