use qf_compiler::{
    QuantumCircuit, // Core structs
    QuantumState,
    hadamard,       // Gates
    rz,
    cnot,
    swap,
};

// A minimal example showing how to implement Quantum Fourier Transform
fn main() -> Result<(), String> {
    println!("--- Minimal QFT Circuit Example ---");

    // --- 1. Define Circuit Parameters ---
    let num_qubits = 3;  // Using 3 qubits for simple demonstration
    let shots = 1000;    // How many times to run the measurement simulation

    println!("Building a {}-qubit QFT circuit.", num_qubits);

    // --- 2. Create an empty Quantum Circuit ---
    let mut circuit = QuantumCircuit::new(num_qubits);

    // --- 3. Add QFT Circuit Elements ---
    println!("Adding QFT gates...");
    
    // Step 1: Apply Hadamard gates and controlled rotations
    for i in 0..num_qubits {
        // Hadamard on current qubit
        circuit.add_gate(hadamard(i));
        
        // Controlled rotations using CNOT and RZ
        for j in 1..num_qubits - i {
            let target = i;
            let control = i + j;
            
            // For controlled phase rotation with angle π/2^j:
            // 1. Apply RZ(π/2^(j+1)) to target
            // 2. CNOT from control to target
            // 3. Apply RZ(-π/2^(j+1)) to target
            // 4. CNOT from control to target again
            let angle = std::f64::consts::PI / 2f64.powf(j as f64);
            
            circuit.add_gate(rz(target, angle / 2.0));
            circuit.add_gate(cnot(control, target));
            circuit.add_gate(rz(target, -angle / 2.0));
            circuit.add_gate(cnot(control, target));
        }
    }
    
    // Step 2: Regular swap qubits for correct output ordering
    for i in 0..num_qubits / 2 {
        let qubit1 = i;
        let qubit2 = num_qubits - 1 - i;
        if qubit1 != qubit2 {
            circuit.add_gate(swap(qubit1, qubit2));
        }
    }

    // --- 4. Simulation & Measurement ---
    println!("Simulating circuit {} times...", shots);
    
    // Create an input state |001⟩
    let mut initial_state = QuantumState::new(num_qubits);
    initial_state.set_computational_basis_state(1)?;
    
    let final_state = circuit.apply_to_state(&initial_state)?;

    println!("\n{}", final_state.measure(shots));
    println!("\n--- Example Finished ---");
    Ok(())
}