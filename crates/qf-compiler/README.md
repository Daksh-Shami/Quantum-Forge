# QF Compiler

The core quantum simulation and execution engine for the Quantum Forge project. This crate provides the foundation for simulating quantum circuits, implementing quantum algorithms, and executing quantum computations.

## Features

- **Quantum Circuit Simulation**: State vector simulation of quantum circuits
- **Modular Architecture**: Extensible design for different simulator backends (CPU, GPU, etc.)
- **Quantum Gates**: Implementation of standard quantum gates (Hadamard, CNOT, etc.)
- **Measurement**: Simulation of quantum measurements with configurable shots
- **Quantum Algorithms**: Implementation of common quantum algorithms
- **Memory Optimization**: Efficient memory usage for large quantum simulations

## Directory Structure

- **src/**: Source code for the quantum compiler
  - **simulator/**: Quantum simulator implementations
    - **cpu.rs**: CPU-based quantum simulator
    - **mod.rs**: Simulator interface and factory functions
  - **algorithms/**: Implementations of quantum algorithms
  - **complex.rs**: Complex number implementation for quantum states
  - **circuit_executor.rs**: Execution engine for quantum circuits
  - **algorithm_runner.rs**: Framework for running quantum algorithms
  - **lib.rs**: Main library file with core types and functions
- **examples/**: Example quantum circuits and algorithms
  - **bell_state.rs**: Implementation of a Bell state circuit
  - **qft.rs**: Quantum Fourier Transform implementation
- **benches/**: Benchmarks for performance testing
- **tests/**: Unit and integration tests

## Usage

### Basic Example: Creating a Bell State

```rust
use qf_compiler::{
    QuantumCircuit, 
    QuantumState,
    cnot, 
    hadamard,
};

fn main() -> Result<(), String> {
    // Create a 2-qubit circuit
    let mut circuit = QuantumCircuit::new(2);
    
    // Add gates to create a Bell state
    circuit.add_gate(hadamard(0));
    circuit.add_gate(cnot(0, 1));
    
    // Simulate the circuit
    let initial_state = QuantumState::new(2);
    let final_state = circuit.apply_to_state(&initial_state)?;
    
    // Measure the state 1000 times
    println!("{}", final_state.measure(1000));
    
    Ok(())
}
```

### Simulator Architecture

The QF Compiler uses a modular simulator architecture that allows for different backend implementations:

1. **Simulator Interface**: The `QuantumSimulator` trait defines the interface for all simulator implementations.
2. **CPU Simulator**: The default implementation uses CPU-based state vector simulation.
3. **Extensibility**: The architecture is designed to be extended with GPU, distributed, or hardware-based simulators.

### Memory Management

Quantum simulation is memory-intensive due to the exponential growth of the state vector. The QF Compiler includes:

- Memory usage tracking
- Optimized data structures for state vectors
- Custom memory allocator (MiMalloc) for improved performance

## Development priorities

The more types of simulations we can experiment with (as a community), the better. I highly encourage you, especially the researchers among you, to implement and experiment with different simulation methods.

At a later date, the maintainers will use this same structure to support real quantum hardware.

### Adding a New Simulator/Simulation method

To add a new simulator backend (e.g., GPU) or a different simulation method (like Unitary, Dense Matrix, Tensor Networks, etc.):

1. Create a new module file (e.g., `gpu.rs`) and implement the simulator:
   ```rust
   pub struct GPUSimulator {
       state_vector: Vec<Complex>,
       num_qubits: usize,
       // Add simulator-specific fields
   }

   impl GPUSimulator {
       pub fn new(num_qubits: usize) -> Self { ... }
       pub fn from_state(amplitudes: Vec<Complex>, num_qubits: usize) -> Self { ... }
   }

   impl QuantumSimulator for GPUSimulator {
       fn apply_gate(&mut self, gate: &QuantumGate) -> Result<(), String> { ... }
       fn measure(&self) -> Result<MeasurementResult, String> { ... }
       fn get_state(&self) -> Result<QuantumState, String> { ... }
       fn num_qubits(&self) -> usize { ... }
       fn memory_estimate(&self) -> usize { ... }
   }
   ```

2. Add the module to `simulator/mod.rs`:
   ```rust
   mod cpu;
   mod gpu;  // Add this line
   pub use cpu::CPUSimulator;
   pub use gpu::GPUSimulator;  // And this line
   ```

3. Add a variant to the `SimulatorType` enum:
   ```rust
   pub enum SimulatorType {
       CPU,
       GPU,  // Add this variant
   }
   ```

4. Update both factory functions to handle the new type:
   ```rust
   pub fn create_simulator(sim_type: SimulatorType, num_qubits: usize) -> Box<dyn QuantumSimulator> {
       match sim_type {
           SimulatorType::CPU => Box::new(CPUSimulator::new(num_qubits)),
           SimulatorType::GPU => Box::new(GPUSimulator::new(num_qubits)),  // Add this arm
       }
   }

   pub fn create_simulator_from_state(...) {
       match sim_type {
           SimulatorType::CPU => Box::new(CPUSimulator::from_state(amplitudes, num_qubits)),
           SimulatorType::GPU => Box::new(GPUSimulator::from_state(amplitudes, num_qubits)),  // Add this arm
       }
   }
   ```

When implementing a new simulator, consider performance factors such as:
- Memory management (especially for GPU/distributed implementations)
- Batching operations where possible
- Optimizing common operations like state vector manipulation
- Using appropriate data structures for the target platform

The rest of the codebase will automatically work with your new simulator because everything uses trait objects (`Box<dyn QuantumSimulator>`).

### Contributing

Contributions to the QF Compiler are welcome! Please refer to the [CONTRIBUTING.md](../../CONTRIBUTING.md) file in the project root for guidelines.

## Future Directions

- GPU-accelerated simulation
- Distributed quantum simulation
- Integration with quantum hardware providers
- Advanced error correction and noise models
- Optimization of quantum circuits
