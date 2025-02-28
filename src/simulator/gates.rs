use crate::Complex;

pub fn hadamard(qubit: usize) -> super::QuantumGate {
    super::QuantumGate::Hadamard(qubit)
}

pub fn cz(control: usize, target: usize) -> super::QuantumGate {
    super::QuantumGate::CZ(control, target)
}

pub fn rx(qubit: usize, angle: f64) -> super::QuantumGate {
    super::QuantumGate::RX(qubit, angle)
}

pub fn rz(qubit: usize, angle: f64) -> super::QuantumGate {
    super::QuantumGate::RZ(qubit, angle)
}

pub fn x(qubit: usize) -> super::QuantumGate {
    super::QuantumGate::X(qubit)
}

pub fn swap(qubit1: usize, qubit2: usize) -> super::QuantumGate {
    super::QuantumGate::Swap(qubit1, qubit2)
}
