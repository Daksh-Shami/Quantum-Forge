use crate::QuantumCircuit;

pub trait Algorithm {
    fn setup(&mut self, num_qubits: usize) -> Result<(), String>;
    fn run(&self, circuit: &mut QuantumCircuit) -> Result<(), String>;
    fn measure(&self, circuit: &QuantumCircuit, qubit: usize) -> Result<bool, String>;
    fn num_qubits(&self) -> usize;
}