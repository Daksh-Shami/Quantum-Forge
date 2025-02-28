use cuda_builder::CudaBuilder;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    CudaBuilder::new("src/simulator/kernels.rs")
        .copy_to("src/simulator/kernels_ptx.rs")
        .build()?;
    Ok(())
}
