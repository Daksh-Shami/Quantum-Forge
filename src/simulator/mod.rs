// Quantum simulator implementations for the quantum_forge library.
// 
// # Adding a New Simulator Implementation
// 
// To add a new simulator type (e.g., GPU, QPU, distributed, etc.), follow these steps:
// 
// 1. Create a new module file (e.g., `gpu.rs`) and implement the simulator:
// ```rust
// pub struct GPUSimulator {
//     state_vector: Vec<Complex>,
//     num_qubits: usize,
//     // Add simulator-specific fields
// }
// 
// impl GPUSimulator {
//     pub fn new(num_qubits: usize) -> Self { ... }
//     pub fn from_state(amplitudes: Vec<Complex>, num_qubits: usize) -> Self { ... }
// }
// 
// impl QuantumSimulator for GPUSimulator {
//     fn apply_gate(&mut self, gate: &QuantumGate) -> Result<(), String> { ... }
//     fn measure(&self) -> Result<MeasurementResult, String> { ... }
//     fn get_state(&self) -> Result<QuantumState, String> { ... }
//     fn num_qubits(&self) -> usize { ... }
//     fn memory_estimate(&self) -> usize { ... }
//     fn clone_box(&self) -> Box<dyn QuantumSimulator> { ... }
// }
// ```
// 
// 2. Add the module to `mod.rs`:
// ```rust
// mod cpu;
// mod gpu;  // Add this line
// pub use cpu::CPUSimulator;
// pub use gpu::GPUSimulator;  // And this line
// ```
// 
// 3. Add a variant to the `SimulatorType` enum:
// ```rust
// pub enum SimulatorType {
//     CPU,
//     GPU,  // Add this variant
// }
// ```
// 
// 4. Update both factory functions to handle the new type:
// ```rust
// pub fn create_simulator(sim_type: SimulatorType, num_qubits: usize) -> Box<dyn QuantumSimulator> {
//     match sim_type {
//         SimulatorType::CPU => Box::new(CPUSimulator::new(num_qubits)),
//         SimulatorType::GPU => Box::new(GPUSimulator::new(num_qubits)),  // Add this arm
//     }
// }
// 
// pub fn create_simulator_from_state(...) {
//     match sim_type {
//         SimulatorType::CPU => Box::new(CPUSimulator::from_state(amplitudes, num_qubits)),
//         SimulatorType::GPU => Box::new(GPUSimulator::from_state(amplitudes, num_qubits)),  // Add this arm
//     }
// }
// ```
// 
// That's it! The rest of the codebase will automatically work with your new simulator
// because everything uses trait objects (`Box<dyn QuantumSimulator>`).
// 
// # Note on Performance
// When implementing a new simulator, consider:
// - Memory management (especially for GPU/distributed implementations)
// - Batching operations where possible
// - Optimizing common operations like state vector manipulation
// - Using appropriate data structures for the target platform

use crate::{QuantumGate, MeasurementResult, QuantumState, Complex};

#[derive(Clone, Copy, Debug)]
pub enum SimulatorType {
    CPU,
    // GPU, // Uncomment when implementing GPU simulator
}

impl Default for SimulatorType {
    fn default() -> Self {
        SimulatorType::CPU
    }
}

/// Trait defining the interface for quantum simulation backends
pub trait QuantumSimulator: Send + Sync {
    /// Apply a quantum gate to the current state
    fn apply_gate(&mut self, gate: &QuantumGate) -> Result<(), String>;
    
    /// Measure the quantum state, returning classical measurement results
    fn measure(&self, measurement_order: &[usize]) -> Result<MeasurementResult, String>;
    
    /// Get the current quantum state
    fn get_state(&self) -> Result<QuantumState, String>;
    
    /// Get number of qubits in the system
    fn num_qubits(&self) -> usize;
    
    /// Estimate memory requirements for this simulator
    fn memory_estimate(&self) -> usize;
}

pub mod cpu;
pub mod gpu;
mod kernels;

pub use cpu::CPUSimulator;
pub use gpu::GPUSimulator;

// Implement From trait for CPUSimulator
impl From<(Vec<Complex>, usize)> for CPUSimulator {
    fn from((amplitudes, num_qubits): (Vec<Complex>, usize)) -> Self {
        CPUSimulator::from_state(amplitudes.into_boxed_slice(), num_qubits)
    }
}

// Factory function to create simulators
pub fn create_simulator(sim_type: SimulatorType, num_qubits: usize) -> Box<dyn QuantumSimulator> {
    match sim_type {
        SimulatorType::CPU => Box::new(CPUSimulator::new(num_qubits)),
        // SimulatorType::GPU => Box::new(GPUSimulator::new(num_qubits)),
    }
}

// Factory function to create simulators from an existing state
pub fn create_simulator_from_state(sim_type: SimulatorType, amplitudes: Vec<Complex>, num_qubits: usize) -> Box<dyn QuantumSimulator> {
    match sim_type {
        SimulatorType::CPU => Box::new(CPUSimulator::from_state(amplitudes.into_boxed_slice(), num_qubits)),
        // SimulatorType::GPU => Box::new(GPUSimulator::from_state(amplitudes, num_qubits)),
    }
}
