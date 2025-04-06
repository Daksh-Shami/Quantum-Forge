# Quantum Forge

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Status](https://img.shields.io/badge/Status-Alpha-orange.svg)
![Rust](https://img.shields.io/badge/Rust-1.71+-orange.svg)
[![Quantum Forge CI](https://github.com/Daksh-Shami/Quantum-Forge/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/Daksh-Shami/Quantum-Forge/actions/workflows/ci.yml)

A high-performance Quantum simulation & visualization framework built in Rust, designed for efficient quantum circuit execution and analysis using intuitive UIs.

## Feature Set for first release (in progress)

- **Open source Compiler w/ OpenQASM 3.0 Support**: Full IR implementation for the latest OpenQASM standard
- **JIT compiler**: Compiles code in real-time using cranelift, thus offering blazing fast iteration for researchers
- **High-Performance Simulation**: Rust-based backend optimized for small-to-medium quantum circuits on any device
- **Interactive Visualization**: Integrated with [Tauri](https://v2.tauri.app/), thus giving us framework to build tools like Circuit building and visualization capabilities.
- **Modular Architecture**: Designed for extensibility and future hardware integration.

## Project Status

Quantum Forge is currently in alpha development. See our [project milestones](../../milestones) for the development roadmap.

Current progress:
- OpenQASM 3 Parsing & AST Processing: Complete
- Basic CPU-based Simulation Backend: Complete
- Intermediate Representation (IR) Implementation: In Progress
- Interactive Circuit Builder: In Progress
- Benchmarking Suite: In Progress

## Getting Started

### Prerequisites

- Rust
- Command Line/Terminal

### Installation

#### Clone the repository
```bash
git clone https://github.com/Daksh-Shami/Quantum-Forge.git && cd Quantum-Forge
```

#### Build the project
```bash
cargo build --release
```
> ⚠️ **WARNING (for Windows):**  
> If your `cargo build` fails and you are on Windows, try running it again in **Developer PowerShell for VS Code 2022**.  
> If you don't have it installed, please install **Visual Studio 2022** first.

#### Run tests
```bash
cargo test
```

#### (Advanced, optional) Run benchmarks (vs. previous run of QF-compiler)
```bash
cargo bench
```
> This option is great for testing the effect of your changes. Whenever you made a change you think should improve the latency a lot, just run `cargo bench` and it will show you how much the performance improved relative to when you started. Key is **you should run this both before AND after making changes**.


### Basic Usage

Quantum Forge provides a clean, intuitive API for building and simulating quantum circuits. Here's a simple example to create a Bell state:

```rust
use qf_compiler::{
    QuantumCircuit, // Core structs
    QuantumState,
    cnot, // Gates
    hadamard,
};

fn main() -> Result<(), String> {
    // Define circuit parameters
    let num_qubits = 2;
    let shots = 1000; // Number of measurement simulations
    
    // Create an empty quantum circuit
    let mut circuit = QuantumCircuit::new(num_qubits);
    
    // Add gates programmatically
    circuit.add_gate(hadamard(0));  // Apply Hadamard to qubit 0
    circuit.add_gate(cnot(0, 1));   // Apply CNOT with control qubit 0, target qubit 1
    
    // Simulation and measurement
    let initial_state = QuantumState::new(num_qubits);
    let final_state = circuit.apply_to_state(&initial_state)?;
    
    // Measure and print results
    println!("{}", final_state.measure(shots));
    // Expect states '00' and '11' with roughly equal probability
    
    Ok(())
}
```

### Examples

Quantum Forge includes pre-built examples to help you get started:

1. **Bell State** - A simple entanglement example:
   ```bash
   cargo run --release --example bell_state
   ```
2. **Quantum Fourier Transform (QFT)** - Implementation of the fundamental QFT algorithm:
   ```bash
   cargo run --release --example qft
   ```
Run these examples to see Quantum Forge in action and as a starting point for your own quantum algorithms.

## Performance

Quantum Forge aims to provide superior performance for small-to-medium scale quantum circuit simulations. Initial benchmarks show promising results against existing frameworks:

![Performance Benchmarks](./assets/benchmark.png)

*Note: These are preliminary benchmarks for just demonstration, and will be expanded in future releases.*

## Roadmap

See our [GitHub Milestones](../../milestones) for detailed development plans.

## Contributing

Contributions are welcome. Please review the [contributing guidelines](CONTRIBUTING.md) before getting started.

### Getting Involved

- Check our [issues labeled "good first issue"](../../issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) to get started.
- See our [help wanted issues](../../issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) for more challenging tasks.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The quantum computing open source community, especially IBM for their OpenQASM language
- All contributors and supporters of this project

---

*Quantum Forge is not affiliated with any commercial quantum computing provider.*
