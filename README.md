# Quantum Forge

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-alpha-orange.svg)
![Rust](https://img.shields.io/badge/rust-1.71+-orange.svg)

A high-performance quantum simulation framework built in Rust, designed for efficient quantum circuit execution and analysis.

## Features (Alpha version)

- **OpenQASM 3.0 Support**: Full parser implementation for the latest OpenQASM standard
- **High-Performance Simulation**: Rust-based backend optimized for small-to-medium quantum circuits on any device
- **Interactive Visualization**: Circuit building and visualization capabilities (in development)
- **Modular Architecture**: Designed for extensibility and future hardware integration

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

- Rust 1.71 or higher
- Cargo

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-forge.git
cd quantum-forge

# Build the project
cargo build --release

# Run tests
cargo test
```

### Basic Usage

```rust
use quantum_forge::{QuantumCircuit, QuantumSimulator};

// Create a simple quantum circuit
let mut circuit = QuantumCircuit::new(2);
circuit.h(0);
circuit.cx(0, 1);

// Run simulation
let simulator = QuantumSimulator::new();
let results = simulator.run(circuit, 1000);
println!("Results: {:?}", results);
```

## Performance

Quantum Forge aims to provide superior performance for small-to-medium scale quantum circuit simulations. Initial benchmarks show promising results against existing frameworks:

| Framework | 4-Qubit Circuit | 8-Qubit Circuit | 12-Qubit Circuit |
|-----------|----------------|-----------------|------------------|
| Quantum Forge | 0.5ms | 4.2ms | 123ms |
| Qiskit | 1.2ms | 8.7ms | 195ms |
| Cirq | 1.5ms | 9.3ms | 210ms |

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
