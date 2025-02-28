use crate::algorithms::algorithm::Algorithm;
use crate::{MeasurementResult, QuantumCircuit, QuantumState};

pub struct AlgorithmRunner<A: Algorithm> {
    algorithm: A,
    circuit: Option<QuantumCircuit>,
    current_state: Option<QuantumState>,
}

impl<A: Algorithm> AlgorithmRunner<A> {
    pub fn new(algorithm: A) -> Self {
        AlgorithmRunner {
            algorithm,
            circuit: None,
            current_state: None,
        }
    }

    pub fn circuit(&self) -> Option<&QuantumCircuit> {
        self.circuit.as_ref()
    }

    pub fn run(&mut self, num_qubits: usize) -> Result<(), String> {
        if num_qubits != self.algorithm.num_qubits() {
            return Err(format!(
                "Number of qubits mismatch. Expected {}, got {}",
                self.algorithm.num_qubits(),
                num_qubits
            ));
        }

        self.algorithm.setup(num_qubits)?;
        let mut circuit = QuantumCircuit::new(num_qubits);
        self.algorithm.run(&mut circuit)?;

        // Create initial state and apply circuit
        let initial_state = QuantumState::new(num_qubits);
        let final_state = circuit.apply_to_state(&initial_state)?;

        // Store both circuit and state
        self.circuit = Some(circuit);
        self.current_state = Some(final_state);

        Ok(())
    }

    pub fn measure(&self, qubit: usize) -> Result<bool, String> {
        match &self.circuit {
            Some(circuit) => self.algorithm.measure(circuit, qubit),
            None => {
                Err("No quantum circuit available. Did you run the algorithm first?".to_string())
            }
        }
    }

    pub fn measure_all(&self, measurement_order: &[usize]) -> Result<MeasurementResult, String> {
        match &self.circuit {
            Some(circuit) => circuit.measureall(measurement_order),
            None => Err("No circuit available to measure".to_string()),
        }
    }
}
