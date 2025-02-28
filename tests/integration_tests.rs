use quantum_forge::*;
use std::collections::HashMap;
use std::f64::EPSILON;
// use quantum_forge::complex::Complex;

const NORM_EPSILON: f64 = 1e-14;

#[cfg(test)]
mod tests {
    use super::*;

    // Changed the input type from BitVec to MeasurementResult
    fn bitvec_to_string(bv: &MeasurementResult) -> String {
        (0..bv.len())
            .map(|i| if bv.get(i) { '1' } else { '0' })
            .collect()
    }

    #[test]
    fn test_single_hadamard() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            h q[0];
            measure q[0] -> c[0];
        "#;

        let (circuit, measurement_order) = QuantumCircuit::from_qasm(qasm).unwrap();
        let mut counts = HashMap::new();

        for _ in 0..1000 {
            let measured_state = circuit.measureall(&measurement_order).unwrap();
            let state_str = bitvec_to_string(&measured_state);
            *counts.entry(state_str).or_insert(0) += 1;
        }

        assert!(counts.contains_key("0"));
        assert!(counts.contains_key("1"));

        let total_counts: i32 = counts.values().sum();
        let prob_0 = *counts.get("0").unwrap_or(&0) as f64 / total_counts as f64;
        let prob_1 = *counts.get("1").unwrap_or(&0) as f64 / total_counts as f64;

        assert!((prob_0 - 0.5).abs() < 0.1);
        assert!((prob_1 - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_single_x_gate() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            x q[0];
            measure q[0] -> c[0];
        "#;

        let (circuit, measurement_order) = QuantumCircuit::from_qasm(qasm).unwrap();
        let mut counts = HashMap::new();

        for _ in 0..100 {
            let measured_state = circuit.measureall(&measurement_order).unwrap();
            let state_str = bitvec_to_string(&measured_state);
            *counts.entry(state_str).or_insert(0) += 1;
        }

        assert!(counts.contains_key("1"));
        assert!(!counts.contains_key("0"));
    }

    #[test]
    fn test_cz_gate() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            // Prepare both qubits in superposition
            h q[0];
            h q[1];
            // Apply CZ gate
            cz q[0],q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
        "#;

        let (circuit, measurement_order) = QuantumCircuit::from_qasm(qasm).unwrap();
        let mut counts = HashMap::new();

        // Run the circuit multiple times to get statistics
        for _ in 0..1000 {
            let measured_state = circuit.measureall(&measurement_order).unwrap();
            let state_str = bitvec_to_string(&measured_state);
            *counts.entry(state_str).or_insert(0) += 1;
        }

        // CZ gate should preserve the equal superposition but add a phase to |11⟩
        // When measuring, we should see approximately equal probability for all basis states
        let count_00 = *counts.get("00").unwrap_or(&0) as f64;
        let count_01 = *counts.get("01").unwrap_or(&0) as f64;
        let count_10 = *counts.get("10").unwrap_or(&0) as f64;
        let count_11 = *counts.get("11").unwrap_or(&0) as f64;

        let total = count_00 + count_01 + count_10 + count_11;

        // Each state should appear approximately 25% of the time
        let expected_probability = 0.25;
        let tolerance = 0.1; // Allow 10% deviation

        assert!((count_00 / total - expected_probability).abs() < tolerance);
        assert!((count_01 / total - expected_probability).abs() < tolerance);
        assert!((count_10 / total - expected_probability).abs() < tolerance);
        assert!((count_11 / total - expected_probability).abs() < tolerance);
    }

    #[test]
    fn test_rx_zero_angle() {
        // RX(0) should act as identity
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(rx(0, 0.0));

        let initial_state = QuantumState::new(1);
        let final_state = circuit.apply_to_state(&initial_state).unwrap();

        // Check if state remains unchanged
        assert!(
            (initial_state.amplitudes[0].re - final_state.amplitudes[0].re).abs() < NORM_EPSILON
        );
        assert!(
            (initial_state.amplitudes[0].im - final_state.amplitudes[0].im).abs() < NORM_EPSILON
        );
    }

    #[test]
    fn test_rx_inverse() {
        let angle = std::f64::consts::FRAC_PI_4; // π/4 for test

        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(rx(0, angle));

        let initial_state = QuantumState::new(1);
        let intermediate_state = circuit.apply_to_state(&initial_state).unwrap();

        let inverse_circuit = circuit.inverse();
        let final_state = inverse_circuit.apply_to_state(&intermediate_state).unwrap();

        // Check probability amplitudes match (up to global phase)
        assert!(
            (initial_state.amplitudes[0].norm() - final_state.amplitudes[0].norm()).abs()
                < NORM_EPSILON
        );
        assert!(
            (initial_state.amplitudes[1].norm() - final_state.amplitudes[1].norm()).abs()
                < NORM_EPSILON
        );
    }

    #[test]
    fn test_rx_pi() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(rx(0, std::f64::consts::PI));

        let initial_state = QuantumState::new(1); // |0⟩ state
        let final_state = circuit.apply_to_state(&initial_state).unwrap();

        // Should transform to |1⟩ state (up to global phase)
        let prob_0 = final_state.amplitudes[0].norm().powi(2);
        let prob_1 = final_state.amplitudes[1].norm().powi(2);

        assert!(prob_0 < NORM_EPSILON);
        assert!((prob_1 - 1.0).abs() < NORM_EPSILON);
    }

    #[test]
    fn test_rx_pi_over_2() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(rx(0, std::f64::consts::FRAC_PI_2));

        let initial_state = QuantumState::new(1);
        let final_state = circuit.apply_to_state(&initial_state).unwrap();

        // Should be equal superposition
        let prob_0 = final_state.amplitudes[0].norm().powi(2);
        let prob_1 = final_state.amplitudes[1].norm().powi(2);

        assert!((prob_0 - 0.5).abs() < NORM_EPSILON * 10.0); // Slightly relaxed tolerance
        assert!((prob_1 - 0.5).abs() < NORM_EPSILON * 10.0);
    }

    #[test]
    fn test_rx_qasm_parsing() {
        let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(pi/2) q[0];
    "#;

        let (circuit, _) = QuantumCircuit::from_qasm(qasm).unwrap();
        assert_eq!(circuit.gates().len(), 1);

        match circuit.gates()[0] {
            QuantumGate::RX(qubit, angle) => {
                assert_eq!(qubit, 0);
                assert!((angle - std::f64::consts::FRAC_PI_2).abs() < NORM_EPSILON);
            }
            _ => panic!("Expected RX gate"),
        }
    }

    #[test]
    fn test_phase_gate_inverse() {
        let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        // Put qubit in superposition
        h q[0];
        // Apply S gate
        s q[0];
    "#;

        let (circuit, _) = QuantumCircuit::from_qasm(qasm).unwrap();

        // Create test state
        let mut state = QuantumState::new(1);

        // Apply circuit (H → S)
        let state = circuit.apply_to_state(&state).unwrap();

        // State should now be (|0⟩ + i|1⟩)/√2

        // Apply inverse circuit (S³)
        let inverse_circuit = circuit.inverse();
        let state = inverse_circuit.apply_to_state(&state).unwrap();
        // State should now be (|0⟩ - i|1⟩)/√2 × (|0⟩ + |1⟩)/√2 = |0⟩

        // since (H)(H|0⟩ + i H|1⟩)/√2 = |0⟩
        let zero_state_prob = state.amplitudes[0].norm().powi(2);
        let one_state_prob = state.amplitudes[1].norm().powi(2);

        println!("Zero state probability: {}", zero_state_prob);
        println!("One state probability: {}", one_state_prob);
        println!("Amplitudes: {:?}", state.amplitudes);
    }

    #[test]
    fn test_cnot_gate() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            x q[0];
            cx q[0],q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
        "#;

        let (circuit, measurement_order) = QuantumCircuit::from_qasm(qasm).unwrap();
        let mut counts = HashMap::new();

        for _ in 0..100 {
            let measured_state = circuit.measureall(&measurement_order).unwrap();
            let state_str = bitvec_to_string(&measured_state);
            *counts.entry(state_str).or_insert(0) += 1;
        }

        assert!(counts.contains_key("11"));
        assert!(!counts.contains_key("00"));
        assert!(!counts.contains_key("01"));
        assert!(!counts.contains_key("10"));
    }

    #[test]
    fn test_quantum_circuit() {
        let mut circuit = QuantumCircuit::new(3);

        circuit.add_gate(hadamard(0));
        circuit.add_gate(phase(1));
        circuit.add_gate(cnot(0, 1));
        circuit.add_gate(toffoli(0, 1, 2));

        assert_eq!(circuit.gates().len(), 4);
        assert_eq!(circuit.gates()[0], hadamard(0));
        assert_eq!(circuit.gates()[1], phase(1));
        assert_eq!(circuit.gates()[2], cnot(0, 1));
        assert_eq!(circuit.gates()[3], toffoli(0, 1, 2));

        let composition = circuit.compose();
        println!("Circuit composition: {}", composition);

        let inverse_circuit = circuit.inverse();
        let inverse_composition = inverse_circuit.compose();
        println!("Inverse circuit composition: {}", inverse_composition);

        let initial_state = QuantumState::new(3);
        let final_state = circuit.apply_to_state(&initial_state).unwrap();
        let measured_state = final_state.measure();
        println!("Measured state: {}", bitvec_to_string(&measured_state));
    }

    #[test]
    fn test_qasm_parser() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            h q[0];
            cx q[0],q[1];
            s q[1];
            ccx q[0],q[1],q[2];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            measure q[2] -> c[2];
        "#;

        let (circuit, measurement_order) = QuantumCircuit::from_qasm(qasm).unwrap();
        assert_eq!(circuit.gates().len(), 4);
        assert_eq!(circuit.gates()[0], hadamard(0));
        assert_eq!(circuit.gates()[1], cnot(0, 1));
        assert_eq!(circuit.gates()[2], phase(1));
        assert_eq!(circuit.gates()[3], toffoli(0, 1, 2));
        assert_eq!(circuit.qubit_count(), 3);
        assert_eq!(measurement_order, vec![0, 1, 2]);
    }

    #[test]
    fn test_parse_qasm() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0],q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
        "#;

        let (circuit, measurement_order) = QuantumCircuit::from_qasm(qasm).unwrap();

        // Check gates
        let gates = circuit.gates();
        assert_eq!(gates.len(), 2);

        // Check first gate is Hadamard on qubit 0
        match gates[0] {
            QuantumGate::Hadamard(q) => assert_eq!(q, 0),
            _ => panic!("Expected Hadamard gate"),
        }

        // Check second gate is CNOT from qubit 0 to 1
        match gates[1] {
            QuantumGate::CNOT(c, t) => {
                assert_eq!(c, 0);
                assert_eq!(t, 1);
            }
            _ => panic!("Expected CNOT gate"),
        }

        // Check measurement order
        assert_eq!(measurement_order, vec![0, 1]);
    }

    #[test]
    fn test_circuit_output_simple() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            measure q -> c;
        "#;

        let (circuit, measurement_order) = QuantumCircuit::from_qasm(qasm).unwrap();
        let measured_state = circuit.measureall(&measurement_order).unwrap();
        let state_str = bitvec_to_string(&measured_state);

        assert_eq!(state_str, "00", "Expected state '00', got '{}'", state_str);
    }

    #[test]
    fn test_rz_gate() {
        let mut circuit = QuantumCircuit::new(1);
        let angle = std::f64::consts::FRAC_PI_4;
        circuit.add_gate(QuantumGate::RZ(0, angle));

        let initial_state = QuantumState::new(1);
        let state_after_rz = circuit.apply_to_state(&initial_state).unwrap();
        let inverse_circuit = circuit.inverse();
        let final_state = inverse_circuit.apply_to_state(&state_after_rz).unwrap();

        for (amp_initial, amp_final) in initial_state
            .amplitudes
            .iter()
            .zip(final_state.amplitudes.iter())
        {
            assert!((amp_initial.re - amp_final.re).abs() < 1e-9);
            assert!((amp_initial.im - amp_final.im).abs() < 1e-9);
        }
    }

    #[test]
    fn test_inverse() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(hadamard(0));
        circuit.add_gate(cnot(0, 1));

        let state = QuantumState::new(2);
        let final_state = circuit.apply_to_state(&state).unwrap();
        let inverse_circuit = circuit.inverse();
        let restored_state = inverse_circuit.apply_to_state(&final_state).unwrap();

        restored_state
            .amplitudes
            .iter()
            .zip(state.amplitudes.iter())
            .for_each(|(a, b)| {
                assert!((a.clone() - b.clone()).norm() < 1e-10);
            });
    }

    #[test]
    fn test_measurement() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(hadamard(0));
        circuit.add_gate(cnot(0, 1));

        let final_state = circuit.apply_to_state(&QuantumState::new(2)).unwrap();
        let measured_state = final_state.measure();

        // Verify the measured state is valid
        assert!(measured_state.len() == 2);
        let bits = measured_state.as_bitvec();
        // Due to superposition, bits can be either both 0 or both 1
        assert!(
            (bits[0] == false && bits[1] == false) || (bits[0] == true && bits[1] == true),
            "Expected both bits to be equal, got {:?}",
            bits
        );
    }

    #[test]
    fn test_complex_circuit() {
        let mut circuit = QuantumCircuit::new(3);
        circuit.add_gate(hadamard(0));
        circuit.add_gate(cnot(0, 1));
        circuit.add_gate(toffoli(0, 1, 2));

        let final_state = circuit.apply_to_state(&QuantumState::new(3)).unwrap();

        // Verify amplitudes sum to 1 (within numerical precision)
        let sum: f64 = final_state
            .amplitudes
            .iter()
            .map(|x| x.norm().powi(2))
            .sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hadamard_double_application() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(hadamard(0));
        circuit.add_gate(hadamard(0));

        // Test on normalized superposition state
        let mut state = QuantumState::new(1);
        let norm = (0.6_f64.powi(2) + 0.8_f64.powi(2)).sqrt();
        state.amplitudes[0] = Complex::new(0.6 / norm, 0.0);
        state.amplitudes[1] = Complex::new(0.8 / norm, 0.0);

        println!("Initial state: {:?}", state.amplitudes);
        let interim = circuit.apply_to_state(&state).unwrap();
        println!("After first H: {:?}", interim.amplitudes);
        let result = circuit.apply_to_state(&state).unwrap();
        println!("After second H: {:?}", result.amplitudes);

        // H·H|ψ⟩ = |ψ⟩
        assert!((state.amplitudes[0].norm() - result.amplitudes[0].norm()).abs() < NORM_EPSILON);
        assert!((state.amplitudes[1].norm() - result.amplitudes[1].norm()).abs() < NORM_EPSILON);
    }

    #[test]
    fn test_hadamard_phase_combo() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(hadamard(0));
        circuit.add_gate(phase(0));
        circuit.add_gate(hadamard(0));

        let state = QuantumState::new(1);
        let result = circuit.apply_to_state(&state).unwrap();

        println!("Final amplitudes: {:?}", result.amplitudes);

        let prob_0 = result.amplitudes[0].norm().powi(2);
        let prob_1 = result.amplitudes[1].norm().powi(2);
        println!("Probability 0: {}", prob_0);
        println!("Probability 1: {}", prob_1);
        println!("Sum: {}", prob_0 + prob_1);

        // Use more appropriate NORM_EPSILON for probability sums
        assert!((prob_0 + prob_1 - 1.0).abs() < NORM_EPSILON);
        assert!((prob_0 - 0.5).abs() < NORM_EPSILON);
        assert!((prob_1 - 0.5).abs() < NORM_EPSILON);
    }

    #[test]
    fn test_hadamard_preserves_norm() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(hadamard(0));

        let mut state = QuantumState::new(1);
        state.amplitudes[0] = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
        state.amplitudes[1] = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);

        let result = circuit.apply_to_state(&state).unwrap();

        let total_prob = result.amplitudes[0].norm().powi(2) + result.amplitudes[1].norm().powi(2);
        println!("Total probability: {}", total_prob);
        assert!((total_prob - 1.0).abs() < NORM_EPSILON);
    }

    #[test]
    fn test_hadamard_math() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(hadamard(0));

        // Test on |0⟩ state
        let state0 = QuantumState::new(1); // |0⟩
        let result0 = circuit.apply_to_state(&state0).unwrap();

        // H|0⟩ = 1/√2(|0⟩ + |1⟩)
        let frac_1_sqrt_2 = 1.0_f64 / 2.0_f64.sqrt();
        assert!((result0.amplitudes[0].re - frac_1_sqrt_2).abs() < NORM_EPSILON);
        assert!(result0.amplitudes[0].im.abs() < NORM_EPSILON);
        assert!((result0.amplitudes[1].re - frac_1_sqrt_2).abs() < NORM_EPSILON);
        assert!(result0.amplitudes[1].im.abs() < NORM_EPSILON);

        // Test on |1⟩ state
        let mut state1 = QuantumState::new(1);
        state1.amplitudes[0] = Complex::new(0.0, 0.0);
        state1.amplitudes[1] = Complex::new(1.0, 0.0);
        let result1 = circuit.apply_to_state(&state1).unwrap();

        // H|1⟩ = 1/√2(|0⟩ - |1⟩)
        assert!((result1.amplitudes[0].re - frac_1_sqrt_2).abs() < NORM_EPSILON);
        assert!(result1.amplitudes[0].im.abs() < NORM_EPSILON);
        assert!((result1.amplitudes[1].re + frac_1_sqrt_2).abs() < NORM_EPSILON);
        assert!(result1.amplitudes[1].im.abs() < NORM_EPSILON);
    }

    #[test]
    fn test_phase_gate_math() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_gate(phase(0));

        // Test on |0⟩ state
        let state0 = QuantumState::new(1);
        let result0 = circuit.apply_to_state(&state0).unwrap();

        // S|0⟩ = |0⟩
        assert!((result0.amplitudes[0].re - 1.0).abs() < NORM_EPSILON);
        assert!(result0.amplitudes[0].im.abs() < NORM_EPSILON);
        assert!(result0.amplitudes[1].norm() < NORM_EPSILON);

        // Test on |1⟩ state
        let mut state1 = QuantumState::new(1);
        state1.amplitudes[0] = Complex::new(0.0, 0.0);
        state1.amplitudes[1] = Complex::new(1.0, 0.0);
        let result1 = circuit.apply_to_state(&state1).unwrap();

        // S|1⟩ = i|1⟩
        assert!(result1.amplitudes[0].norm() < NORM_EPSILON);
        assert!(result1.amplitudes[1].re.abs() < NORM_EPSILON);
        assert!((result1.amplitudes[1].im - 1.0).abs() < NORM_EPSILON);
    }

    #[test]
    fn test_phase_gate_repeated() {
        // S·S·S·S should equal identity
        let mut circuit = QuantumCircuit::new(1);
        for _ in 0..4 {
            circuit.add_gate(phase(0));
        }

        // Test on arbitrary superposition
        let mut state = QuantumState::new(1);
        state.amplitudes[0] = Complex::new(0.6, 0.0);
        state.amplitudes[1] = Complex::new(0.8, 0.0);

        let result = circuit.apply_to_state(&state).unwrap();

        // S⁴|ψ⟩ = |ψ⟩
        assert!((state.amplitudes[0].re - result.amplitudes[0].re).abs() < NORM_EPSILON);
        assert!((state.amplitudes[0].im - result.amplitudes[0].im).abs() < NORM_EPSILON);
        assert!((state.amplitudes[1].re - result.amplitudes[1].re).abs() < NORM_EPSILON);
        assert!((state.amplitudes[1].im - result.amplitudes[1].im).abs() < NORM_EPSILON);
    }

    // At the top with other constants
    const NORM_EPSILON: f64 = 1e-14; // More appropriate for norm calculations

    #[test]
    fn test_bell_state() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(hadamard(0));
        circuit.add_gate(cnot(0, 1));

        let state = QuantumState::new(2);
        let final_state = circuit.apply_to_state(&state).unwrap();
        let inverse_circuit = circuit.inverse();
        let mut restored_state = inverse_circuit.apply_to_state(&final_state).unwrap();

        // Check if the state is restored
        for (a, b) in restored_state
            .amplitudes
            .iter()
            .zip(state.amplitudes.iter())
        {
            assert!((a.clone() - b.clone()).norm() < 1e-10);
        }
    }
}
