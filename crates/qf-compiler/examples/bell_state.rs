use qf_compiler::{
    QuantumCircuit, // Core structs
    QuantumState,
    cnot, // Gates
    hadamard,
};

// A minimal example showing how to build and simulate a circuit programmatically.
fn main() -> Result<(), String> {
    println!("--- Minimal Programmatic Circuit Example ---");

    // --- 1. Define Circuit Parameters ---
    let num_qubits = 2;
    let shots = 1000; // How many times to run the measurement simulation

    println!("Building a {}-qubit Bell state circuit.", num_qubits);

    // --- 2. Create an empty Quantum Circuit ---
    let mut circuit = QuantumCircuit::new(num_qubits);

    // --- 3. Add Gates Programmatically ---
    println!("Adding gates: H(0), CNOT(0, 1)");
    circuit.add_gate(hadamard(0));
    circuit.add_gate(cnot(0, 1));

    // --- 4. Simulation & Measurement ---
    println!("Simulating circuit {} times...", shots);
    let initial_state = QuantumState::new(num_qubits);
    let final_state = circuit.apply_to_state(&initial_state)?;

    println!("\n{}", final_state.measure(shots));
    println!("\n(Expect states '00' and '11' with roughly equal probability)");
    println!("--- Example Finished ---");
    Ok(())
}
