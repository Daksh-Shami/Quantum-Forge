use criterion::{black_box, criterion_group, criterion_main, Criterion};
use qf_compiler::*;
use std::time::Duration;

macro_rules! bench_circuit {
    ($c:expr, $qubits:expr) => {
        let qasm = format!(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[{}];
            creg c[{}];
            {}
            measure q -> c;
            "#,
            $qubits,
            $qubits,
            generate_complex_circuit($qubits)
        );
        let (circuit, measurement_order) = QuantumCircuit::from_qasm(&qasm).unwrap();

        $c.bench_function(&format!("{} qubit circuit", $qubits), |b| {
            b.iter(|| circuit.measureall(black_box(&measurement_order)))
        });
    };
}

fn generate_complex_circuit(qubits: usize) -> String {
    let mut circuit = String::new();
    let layers = 5; // Number of repetitions of the pattern

    for _ in 0..layers {
        // Apply Hadamard gates to all qubits
        for i in 0..qubits {
            circuit.push_str(&format!("h q[{}];\n", i));
        }

        // Apply CNOT gates in a chain
        for i in 0..qubits - 1 {
            circuit.push_str(&format!("cx q[{}],q[{}];\n", i, i + 1));
        }

        // Apply Phase gates to even qubits
        for i in (0..qubits).step_by(2) {
            circuit.push_str(&format!("s q[{}];\n", i));
        }

        // Apply Toffoli gates where possible
        for i in 0..qubits - 2 {
            circuit.push_str(&format!("ccx q[{}],q[{}],q[{}];\n", i, i + 1, i + 2));
        }

        // Apply X gates to odd qubits
        for i in (1..qubits).step_by(2) {
            circuit.push_str(&format!("x q[{}];\n", i));
        }
    }

    circuit
}

fn bench_circuits(c: &mut Criterion) {
    let qubit_counts = [5, 10, 15, 20];

    for &qubits in &qubit_counts {
        bench_circuit!(c, qubits);
    }
}

fn custom_criterion() -> Criterion {
    Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::new(30, 0))
        .configure_from_args()
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = bench_circuits
}
criterion_main!(benches);
