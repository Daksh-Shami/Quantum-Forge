use super::QuantumSimulator;
use crate::{BitVec, Complex, MeasurementResult, QuantumGate, QuantumState};
use aligned_vec::AVec;
use rand::Rng;

#[derive(Clone)]
pub struct CPUSimulator {
    state_vector: AVec<Complex>,
    num_qubits: usize,
}

impl CPUSimulator {
    pub fn new(num_qubits: usize) -> Self {
        let size = 1 << num_qubits;
        let mut state_vector = AVec::new(32); // 32-byte alignment for better cache performance
        state_vector.resize(size, Complex::new(0.0, 0.0));
        state_vector[0] = Complex::new(1.0, 0.0);

        Self {
            state_vector,
            num_qubits,
        }
    }

    pub fn from_state(state_vector: Vec<Complex>, num_qubits: usize) -> Self {
        let mut aligned_vec = AVec::new(32);
        aligned_vec.extend_from_slice(&state_vector);

        Self {
            state_vector: aligned_vec,
            num_qubits,
        }
    }

    fn apply_hadamard(&mut self, qubit: usize) -> Result<(), String> {
        if qubit >= self.num_qubits {
            return Err(format!("Qubit index {} out of range", qubit));
        }

        let factor = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let len = self.state_vector.len();
        let mask = 1 << qubit;

        for i in 0..len {
            if i & mask == 0 {
                let i2 = i | mask;
                if i2 < len {
                    let v1 = self.state_vector[i];
                    let v2 = self.state_vector[i2];

                    let sum = v1 + v2;
                    let diff = v1 - v2;

                    self.state_vector[i] = factor * sum;
                    self.state_vector[i2] = factor * diff;
                }
            }
        }
        Ok(())
    }

    fn apply_rx(&mut self, qubit: usize, angle: f64) -> Result<(), String> {
        if qubit >= self.num_qubits {
            return Err(format!("Qubit index {} out of range", qubit));
        }

        let theta = angle / 2.0;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let len = self.state_vector.len();
        let mask = 1 << qubit;

        for i in 0..len {
            if i & mask == 0 {
                let i2 = i | mask;
                if i2 < len {
                    let v1 = self.state_vector[i];
                    let v2 = self.state_vector[i2];

                    // RX(θ) = [cos(θ/2)    -i*sin(θ/2)]
                    //         [-i*sin(θ/2)   cos(θ/2)  ]
                    self.state_vector[i] = Complex::new(
                        v1.re * cos_theta - v2.im * sin_theta,
                        v1.im * cos_theta + v2.re * sin_theta,
                    );
                    self.state_vector[i2] = Complex::new(
                        v2.re * cos_theta - v1.im * sin_theta,
                        v2.im * cos_theta + v1.re * sin_theta,
                    );
                }
            }
        }
        Ok(())
    }

    fn apply_rz(&mut self, qubit: usize, angle: f64) -> Result<(), String> {
        if qubit >= self.num_qubits {
            return Err(format!("Qubit index {} out of range", qubit));
        }

        let theta = angle / 2.0;
        let rotation = Complex::new(theta.cos(), -theta.sin());

        for i in 0..self.state_vector.len() {
            if i & (1 << qubit) != 0 {
                self.state_vector[i] *= rotation;
            }
        }
        Ok(())
    }

    fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), String> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(format!("Qubit indices out of range"));
        }

        // Only process indices where control bit is 1 and haven't been processed yet
        for i in 0..self.state_vector.len() {
            if (i & (1 << control) != 0) && (i & (1 << target) == 0) {
                let i2 = i ^ (1 << target);
                self.state_vector.swap(i, i2);
            }
        }
        Ok(())
    }

    fn apply_x(&mut self, qubit: usize) -> Result<(), String> {
        if qubit >= self.num_qubits {
            return Err(format!("Qubit index {} out of range", qubit));
        }

        for i in 0..self.state_vector.len() {
            if i & (1 << qubit) == 0 {
                let i2 = i ^ (1 << qubit);
                self.state_vector.swap(i, i2);
            }
        }
        Ok(())
    }

    fn apply_cz(&mut self, control: usize, target: usize) -> Result<(), String> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(format!("Qubit indices out of range"));
        }

        for i in 0..self.state_vector.len() {
            if (i & (1 << control) != 0) && (i & (1 << target) != 0) {
                self.state_vector[i] = -self.state_vector[i];
            }
        }
        Ok(())
    }

    fn apply_toffoli(
        &mut self,
        control1: usize,
        control2: usize,
        target: usize,
    ) -> Result<(), String> {
        if control1 >= self.num_qubits || control2 >= self.num_qubits || target >= self.num_qubits {
            return Err(format!("Qubit indices out of range"));
        }

        for i in 0..self.state_vector.len() {
            if (i & (1 << control1) != 0) && (i & (1 << control2) != 0) {
                let i2 = i ^ (1 << target);
                self.state_vector.swap(i, i2);
            }
        }
        Ok(())
    }

    fn apply_swap(&mut self, qubit1: usize, qubit2: usize) -> Result<(), String> {
        if qubit1 >= self.num_qubits || qubit2 >= self.num_qubits {
            return Err(format!("Qubit indices out of range"));
        }

        for i in 0..self.state_vector.len() {
            if (i & (1 << qubit1) != 0) != (i & (1 << qubit2) != 0) {
                let j = i ^ (1 << qubit1) ^ (1 << qubit2);
                self.state_vector.swap(i, j);
            }
        }
        Ok(())
    }

    fn apply_phase(&mut self, qubit: usize) -> Result<(), String> {
        if qubit >= self.num_qubits {
            return Err(format!("Qubit index {} out of range", qubit));
        }

        let phase = Complex::new(0.0, 1.0);
        let mask = 1 << qubit;

        // Only iterate over states where the qubit is 1
        for i in 0..self.state_vector.len() {
            if i & mask != 0 {
                self.state_vector[i] *= phase;
            }
        }

        Ok(())
    }
}

impl QuantumSimulator for CPUSimulator {
    fn apply_gate(&mut self, gate: &QuantumGate) -> Result<(), String> {
        match gate {
            QuantumGate::Hadamard(qubit) => self.apply_hadamard(*qubit),
            QuantumGate::CNOT(control, target) => self.apply_cnot(*control, *target),
            QuantumGate::X(qubit) => self.apply_x(*qubit),
            QuantumGate::Phase(qubit) => self.apply_phase(*qubit),
            QuantumGate::RZ(qubit, angle) => self.apply_rz(*qubit, *angle),
            QuantumGate::RX(qubit, angle) => self.apply_rx(*qubit, *angle),
            QuantumGate::CZ(control, target) => self.apply_cz(*control, *target),
            QuantumGate::Toffoli(c1, c2, target) => self.apply_toffoli(*c1, *c2, *target),
            QuantumGate::Swap(q1, q2) => self.apply_swap(*q1, *q2),
        }
    }

    fn measure(&self) -> Result<MeasurementResult, String> {
        let mut rng = rand::thread_rng();
        let mut cumulative_prob = 0.0;
        let r = rng.gen::<f64>();

        for (i, amplitude) in self.state_vector.iter().enumerate() {
            cumulative_prob += amplitude.norm().powi(2);
            if r <= cumulative_prob {
                let mut result = BitVec::zeros(self.num_qubits);
                for j in 0..self.num_qubits {
                    result.set(j, (i >> j) & 1 == 1);
                }
                return Ok(MeasurementResult::new(result));
            }
        }

        Err("Measurement failed - probabilities did not sum to 1".to_string())
    }

    fn get_state(&self) -> Result<QuantumState, String> {
        Ok(QuantumState::from_amplitudes_ref(&self.state_vector))
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn memory_estimate(&self) -> usize {
        // Each complex number uses 16 bytes (2 * f64)
        self.state_vector.len() * 16
    }
}
