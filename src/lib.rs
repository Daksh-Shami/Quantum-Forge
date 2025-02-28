use crate::simulator::{create_simulator_from_state, SimulatorType};
use bitvec_simd::BitVec;
use memory_stats::memory_stats;
use mimalloc::MiMalloc;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::f64::EPSILON;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
pub static PEAK_MEMORY: AtomicUsize = AtomicUsize::new(0);

// Do not remove -- This adds algorithms module as a part of the core compiler.
pub mod algorithm_runner;
pub mod algorithms;
pub mod circuit_executor;
// pub mod parser;
pub mod complex;
pub use complex::Complex;

pub mod simulator;
use simulator::QuantumSimulator;

// Optimized bit operations
#[inline(always)]
const fn bit_mask(qubit: usize) -> usize {
    1 << qubit
}

#[derive(Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub struct GroupElement {
    pub symbol: String,
}

impl GroupElement {
    pub fn new(symbol: String) -> Self {
        GroupElement { symbol }
    }
}

impl fmt::Display for GroupElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.symbol)
    }
}

// Keep Debug here for proper testing.
#[derive(Clone, Debug)]
pub enum QuantumGate {
    Hadamard(usize),
    Phase(usize),
    CNOT(usize, usize),
    Toffoli(usize, usize, usize),
    Swap(usize, usize),
    X(usize),
    RZ(usize, f64),
    RX(usize, f64),
    CZ(usize, usize),
}

impl PartialEq for QuantumGate {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (QuantumGate::Hadamard(q1), QuantumGate::Hadamard(q2)) => q1 == q2,
            (QuantumGate::Phase(q1), QuantumGate::Phase(q2)) => q1 == q2,
            (QuantumGate::CNOT(c1, t1), QuantumGate::CNOT(c2, t2)) => c1 == c2 && t1 == t2,
            (QuantumGate::Toffoli(c1, c2, t1), QuantumGate::Toffoli(c3, c4, t2)) => {
                c1 == c3 && c2 == c4 && t1 == t2
            }
            (QuantumGate::Swap(q1, q2), QuantumGate::Swap(q3, q4)) => q1 == q3 && q2 == q4,
            (QuantumGate::X(q1), QuantumGate::X(q2)) => q1 == q2,
            (QuantumGate::RZ(q1, angle1), QuantumGate::RZ(q2, angle2)) => {
                q1 == q2 && (angle1 - angle2).abs() < EPSILON // Approximate equality for RZ
            }
            (QuantumGate::RX(q1, angle1), QuantumGate::RX(q2, angle2)) => {
                q1 == q2 && (angle1 - angle2).abs() < EPSILON
            }
            (QuantumGate::CZ(c1, t1), QuantumGate::CZ(c2, t2)) => c1 == c2 && t1 == t2,
            _ => false,
        }
    }
}

impl Eq for QuantumGate {}

impl QuantumGate {
    pub fn to_qasm(&self) -> String {
        match self {
            QuantumGate::Hadamard(q) => format!("h q[{}];", q),
            QuantumGate::Phase(q) => format!("s q[{}];", q),
            QuantumGate::CNOT(c, t) => format!("cx q[{}],q[{}];", c, t),
            QuantumGate::Toffoli(c1, c2, t) => format!("ccx q[{}],q[{}],q[{}];", c1, c2, t),
            QuantumGate::X(q) => format!("x q[{}];", q),
            QuantumGate::RZ(q, angle) => format!("rz({}) q[{}];", angle, q),
            QuantumGate::RX(q, angle) => format!("rx({}) q[{}];", angle, q),
            QuantumGate::CZ(c, t) => format!("cz q[{}],q[{}];", c, t),
            QuantumGate::Swap(q1, q2) => format!("swap q[{}],q[{}];", q1, q2),
        }
    }
}

#[derive(Clone)]
pub struct QuantumCircuit {
    gates: Vec<QuantumGate>,
    qubit_count: usize,
    classical_registers: Vec<BitVec>,
    simulator_type: SimulatorType,
}

impl QuantumCircuit {
    pub fn new(qubit_count: usize) -> Self {
        Self {
            gates: Vec::new(),
            qubit_count,
            classical_registers: Vec::new(),
            simulator_type: SimulatorType::default(),
        }
    }

    pub fn with_simulator_type(qubit_count: usize, sim_type: SimulatorType) -> Self {
        Self {
            gates: Vec::new(),
            qubit_count,
            classical_registers: Vec::new(),
            simulator_type: sim_type,
        }
    }

    pub fn from_qasm(input: &str) -> Result<(Self, Vec<usize>), String> {
        let instructions = parse_qasm(input)?;
        let mut circuit = None;
        let mut measurement_order = Vec::new();
        let mut qreg_map: HashMap<String, usize> = HashMap::default();
        let mut creg_map: HashMap<String, usize> = HashMap::default();

        for instruction in instructions {
            match instruction {
                QASMInstruction::QReg(name, size) => {
                    circuit = Some(Self::with_simulator_type(size, SimulatorType::default()));
                    qreg_map.insert(name, 0); // Store the register start index
                }
                QASMInstruction::CReg(name, size) => {
                    if let Some(ref mut c) = circuit {
                        let reg_index = c.classical_registers.len();
                        c.add_classical_register(name.clone(), size);
                        creg_map.insert(name, reg_index);
                    }
                }
                QASMInstruction::Hadamard(_reg, idx) => {
                    if let Some(ref mut c) = circuit {
                        c.add_gate(hadamard(idx));
                    }
                }
                QASMInstruction::Phase(_reg, idx) => {
                    if let Some(ref mut c) = circuit {
                        c.add_gate(phase(idx));
                    }
                }
                QASMInstruction::CNOT(_creg, cidx, _treg, tidx) => {
                    if let Some(ref mut c) = circuit {
                        c.add_gate(cnot(cidx, tidx));
                    }
                }
                QASMInstruction::Toffoli(_c1reg, c1idx, _c2reg, c2idx, _treg, tidx) => {
                    if let Some(ref mut c) = circuit {
                        c.add_gate(toffoli(c1idx, c2idx, tidx));
                    }
                }
                QASMInstruction::X(_reg, idx) => {
                    if let Some(ref mut c) = circuit {
                        c.add_gate(x(idx));
                    }
                }
                QASMInstruction::RZ(_reg, idx, angle) => {
                    if let Some(ref mut c) = circuit {
                        c.add_gate(rz(idx, angle));
                    }
                }
                QASMInstruction::RX(_reg, idx, angle) => {
                    if let Some(ref mut c) = circuit {
                        c.add_gate(rx(idx, angle));
                    }
                }
                QASMInstruction::CZ(_creg, cidx, _treg, tidx) => {
                    if let Some(ref mut c) = circuit {
                        c.add_gate(cz(cidx, tidx));
                    }
                }
                QASMInstruction::Swap(_reg1, idx1, _reg2, idx2) => {
                    if let Some(ref mut c) = circuit {
                        c.add_gate(swap(idx1, idx2));
                    }
                }
                QASMInstruction::Measure(_qreg, qidx, _creg, _cidx) => {
                    measurement_order.push(qidx);
                }
                QASMInstruction::MeasureAll => {
                    // Handle measureall instruction if needed
                }
                QASMInstruction::Reset(_reg, _idx) => {
                    // Handle reset instruction if needed
                }
            }
        }

        circuit
            .ok_or_else(|| "No quantum register defined".to_string())
            .map(|c| (c, measurement_order))
    }

    pub fn add_gate(&mut self, gate: QuantumGate) {
        self.gates.push(gate);
    }

    pub fn add_classical_register(&mut self, _name: String, size: usize) {
        self.classical_registers.push(BitVec::zeros(size));
    }

    pub fn get_classical_register(&self, index: usize) -> Option<&BitVec> {
        self.classical_registers.get(index)
    }

    pub fn gates(&self) -> &[QuantumGate] {
        &self.gates
    }

    pub fn qubit_count(&self) -> usize {
        self.qubit_count
    }

    pub fn compose(&self) -> String {
        let mut result = String::new();
        result.push_str(&format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\n"));
        result.push_str(&format!("qreg q[{}];\n", self.qubit_count));

        for (i, _) in self.classical_registers.iter().enumerate() {
            result.push_str(&format!("creg c{}[{}];\n", i, self.qubit_count));
        }

        for gate in &self.gates {
            result.push_str(&format!("{}\n", gate.to_qasm()));
        }

        result
    }

    pub fn inverse(&self) -> Self {
        let mut inverse_circuit = Self::new(self.qubit_count);

        // Add gates in reverse order with their inverse operations
        for gate in self.gates.iter().rev() {
            match gate {
                QuantumGate::Phase(q) => {
                    // Phase gate inverse is conjugate
                    inverse_circuit.add_gate(QuantumGate::Phase(*q));
                    inverse_circuit.add_gate(QuantumGate::Phase(*q));
                    inverse_circuit.add_gate(QuantumGate::Phase(*q));
                }
                QuantumGate::RZ(q, angle) => {
                    inverse_circuit.add_gate(QuantumGate::RZ(*q, -angle));
                }
                QuantumGate::RX(q, angle) => {
                    inverse_circuit.add_gate(QuantumGate::RX(*q, -angle));
                }
                // Self-inverse gates
                _ => inverse_circuit.add_gate(gate.clone()),
            }
        }

        inverse_circuit
    }

    pub fn apply_to_state(&self, initial_state: &QuantumState) -> Result<QuantumState, String> {
        let mut simulator = create_simulator_from_state(
            self.simulator_type,
            initial_state.amplitudes.clone(),
            initial_state.qubit_count,
        );

        for gate in &self.gates {
            simulator.apply_gate(gate)?;
        }

        simulator.get_state()
    }

    pub fn measureall(&self, measurement_order: &[usize]) -> Result<MeasurementResult, String> {
        let state = self.apply_to_state(&QuantumState::new(self.qubit_count))?;
        let simulator =
            create_simulator_from_state(self.simulator_type, state.amplitudes, state.qubit_count);
        simulator.measure(measurement_order)
    }

    pub fn update_classical_registers(
        &mut self,
        measured_state: &MeasurementResult,
        measurement_order: &[usize],
    ) {
        for (i, &qubit) in measurement_order.iter().enumerate() {
            if let Some(reg) = self.classical_registers.get_mut(i) {
                reg.set(qubit, measured_state.get(qubit));
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QuantumState {
    pub amplitudes: Vec<Complex>,
    pub qubit_count: usize,
}

impl QuantumState {
    pub fn new(qubit_count: usize) -> Self {
        let mut amplitudes = vec![Complex::new(0.0, 0.0); 1 << qubit_count];
        amplitudes[0] = Complex::new(1.0, 0.0);
        QuantumState {
            amplitudes,
            qubit_count,
        }
    }

    pub fn from_amplitudes(amplitudes: Vec<Complex>) -> Self {
        let qubit_count = (amplitudes.len() as f64).log2() as usize;
        Self {
            amplitudes,
            qubit_count,
        }
    }

    pub fn from_amplitudes_ref(amplitudes: &[Complex]) -> Self {
        let qubit_count = (amplitudes.len() as f64).log2() as usize;
        Self {
            amplitudes: amplitudes.to_vec(),
            qubit_count,
        }
    }

    pub fn to_simulator<S: QuantumSimulator + From<(Vec<Complex>, usize)>>(
        &self,
    ) -> Result<S, String> {
        Ok(S::from((self.amplitudes.clone(), self.qubit_count)))
    }

    #[inline]
    fn index_to_bitvec(index: usize, size: usize) -> BitVec {
        let mut bv = BitVec::zeros(size);
        for i in 0..size {
            if (index & (1 << i)) != 0 {
                bv.set(i, true);
            }
        }
        bv
    }

    pub fn measure(&self) -> MeasurementResult {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let mut cumulative_prob = 0.0;

        for (i, amplitude) in self.amplitudes.iter().enumerate() {
            let prob = amplitude.norm().powi(2);
            cumulative_prob += prob;

            if r <= cumulative_prob {
                return MeasurementResult::new(Self::index_to_bitvec(i, self.qubit_count));
            }
        }

        MeasurementResult::new(BitVec::ones(self.qubit_count))
    }

    pub fn measure_qubit(&self, index: usize) -> bool {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let mut prob_one = 0.0;

        // Calculate probability of measuring |1⟩ for this qubit
        for (i, amplitude) in self.amplitudes.iter().enumerate() {
            if (i & bit_mask(index)) != 0 {
                // Here
                prob_one += amplitude.norm().powi(2);
            }
        }

        r <= prob_one
    }
}

#[derive(Clone)]
pub struct MeasurementResult(BitVec);

impl MeasurementResult {
    pub fn new(bv: BitVec) -> Self {
        MeasurementResult(bv)
    }

    pub fn get(&self, index: usize) -> bool {
        self.0.get(index).unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn as_bitvec(&self) -> &BitVec {
        &self.0
    }
}

impl Hash for MeasurementResult {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);

        for i in (0..self.0.len()).step_by(64) {
            let mut chunk: u64 = 0;
            for j in 0..64.min(self.0.len() - i) {
                if self.0.get(i + j).unwrap_or(false) {
                    chunk |= 1 << j;
                }
            }
            chunk.hash(state);
        }
    }
}

impl PartialEq for MeasurementResult {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for MeasurementResult {}

pub fn hadamard(qubit: usize) -> QuantumGate {
    QuantumGate::Hadamard(qubit)
}

pub const fn phase(qubit: usize) -> QuantumGate {
    QuantumGate::Phase(qubit)
}

pub const fn cnot(control: usize, target: usize) -> QuantumGate {
    QuantumGate::CNOT(control, target)
}

pub const fn toffoli(control1: usize, control2: usize, target: usize) -> QuantumGate {
    QuantumGate::Toffoli(control1, control2, target)
}

pub fn x(qubit: usize) -> QuantumGate {
    QuantumGate::X(qubit)
}

pub fn rz(qubit: usize, angle: f64) -> QuantumGate {
    QuantumGate::RZ(qubit, angle)
}

pub fn rx(qubit: usize, angle: f64) -> QuantumGate {
    QuantumGate::RX(qubit, angle)
}

pub const fn cz(control: usize, target: usize) -> QuantumGate {
    QuantumGate::CZ(control, target)
}

pub const fn swap(qubit1: usize, qubit2: usize) -> QuantumGate {
    QuantumGate::Swap(qubit1, qubit2)
}

#[derive(Debug, PartialEq)]
pub enum QASMInstruction {
    QReg(String, usize),
    CReg(String, usize),
    Hadamard(String, usize),
    X(String, usize),
    Phase(String, usize),
    CNOT(String, usize, String, usize),
    Toffoli(String, usize, String, usize, String, usize),
    Swap(String, usize, String, usize),
    RZ(String, usize, f64),
    RX(String, usize, f64),
    CZ(String, usize, String, usize),
    Measure(String, usize, String, usize),
    MeasureAll,
    Reset(String, usize),
}

macro_rules! parse_register_info {
    ($reg_info:expr) => {{
        let reg_info = $reg_info.trim_matches(|c| c == ';' || c == '[' || c == ']');
        let (name, size) = reg_info.split_once('[').ok_or("Invalid register format")?;
        let size = size
            .trim_end_matches(']')
            .parse()
            .map_err(|e| format!("Invalid register size: {}", e))?;
        (name.to_string(), size)
    }};
}

fn parse_angle(angle_str: &str) -> Result<f64, String> {
    let angle_str = angle_str.trim();

    // Handle negative pi cases first
    if angle_str.starts_with("-pi/") {
        return angle_str[4..]
            .parse::<f64>()
            .map(|divisor| -PI / divisor)
            .map_err(|e| format!("Invalid divisor in angle: {}", e));
    }

    match angle_str {
        "pi" => Ok(PI),
        "-pi" => Ok(-PI),
        s if s.starts_with("pi/") => s[3..]
            .parse::<f64>()
            .map(|divisor| PI / divisor)
            .map_err(|e| format!("Invalid divisor in angle: {}", e)),
        s => s
            .parse()
            .map_err(|e| format!("Invalid angle format: {}", e)),
    }
}

fn parse_qubit(s: &str) -> Result<(String, usize), String> {
    let s = s.trim();
    if let Some(bracket_index) = s.find('[') {
        let reg = s[..bracket_index].to_string();
        let index_str = &s[bracket_index + 1..];
        let index = index_str
            .trim_end_matches(']')
            .parse()
            .map_err(|e| format!("Invalid qubit index: {}", e))?;
        Ok((reg, index))
    } else {
        Ok((s.to_string(), 0))
    }
}

impl FromStr for QASMInstruction {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split_whitespace().collect();

        // First check if it's a rotation gate by looking at first two characters
        if parts[0].starts_with("rx") || parts[0].starts_with("rz") {
            let gate_type = &parts[0][..2];

            // Extract angle part
            let angle_str = parts[0]
                .find('(')
                .and_then(|start| parts[0].find(')').map(|end| &parts[0][start + 1..end]))
                .ok_or_else(|| format!("Invalid rotation gate format: {}", parts[0]))?;

            let angle = parse_angle(angle_str)?;
            let (reg, index) = parse_qubit(parts[1])?;

            match gate_type {
                "rz" => Ok(QASMInstruction::RZ(reg, index, angle)),
                "rx" => Ok(QASMInstruction::RX(reg, index, angle)),
                _ => unreachable!(),
            }
        } else {
            match parts[0] {
                "qreg" => {
                    let (name, size) = parse_register_info!(parts[1]);
                    Ok(QASMInstruction::QReg(name, size))
                }
                "creg" => {
                    let (name, size) = parse_register_info!(parts[1]);
                    Ok(QASMInstruction::CReg(name, size))
                }
                "x" => {
                    let (reg, index) = parse_qubit(parts[1])?;
                    Ok(QASMInstruction::X(reg, index))
                }
                "h" => {
                    let (reg, index) = parse_qubit(parts[1])?;
                    Ok(QASMInstruction::Hadamard(reg, index))
                }
                "s" => {
                    let (reg, index) = parse_qubit(parts[1])?;
                    Ok(QASMInstruction::Phase(reg, index))
                }
                "cx" => {
                    let args = parts[1..].join("");
                    let qubits: Vec<&str> = args.split(',').collect();
                    if qubits.len() != 2 {
                        return Err(format!("Invalid CNOT format: {}", s));
                    }
                    let (control_reg, control_index) = parse_qubit(qubits[0])?;
                    let (target_reg, target_index) = parse_qubit(qubits[1])?;
                    Ok(QASMInstruction::CNOT(
                        control_reg,
                        control_index,
                        target_reg,
                        target_index,
                    ))
                }
                "ccx" => {
                    let args = parts[1..].join("");
                    let qubits: Vec<&str> = args.split(',').collect();
                    if qubits.len() != 3 {
                        return Err(format!("Invalid Toffoli format: {}", s));
                    }
                    let (control1_reg, control1_index) = parse_qubit(qubits[0])?;
                    let (control2_reg, control2_index) = parse_qubit(qubits[1])?;
                    let (target_reg, target_index) = parse_qubit(qubits[2])?;
                    Ok(QASMInstruction::Toffoli(
                        control1_reg,
                        control1_index,
                        control2_reg,
                        control2_index,
                        target_reg,
                        target_index,
                    ))
                }
                "swap" => {
                    let args = parts[1..].join("");
                    let qubits: Vec<&str> = args.split(',').collect();
                    if qubits.len() != 2 {
                        return Err(format!("Invalid SWAP format: {}", s));
                    }
                    let (q1_reg, q1_index) = parse_qubit(qubits[0])?;
                    let (q2_reg, q2_index) = parse_qubit(qubits[1])?;
                    Ok(QASMInstruction::Swap(q1_reg, q1_index, q2_reg, q2_index))
                }
                "cz" => {
                    let args = parts[1..].join("");
                    let qubits: Vec<&str> = args.split(',').collect();
                    if qubits.len() != 2 {
                        return Err(format!("Invalid CZ format: {}", s));
                    }
                    let (control_reg, control_index) = parse_qubit(qubits[0])?;
                    let (target_reg, target_index) = parse_qubit(qubits[1])?;
                    Ok(QASMInstruction::CZ(
                        control_reg,
                        control_index,
                        target_reg,
                        target_index,
                    ))
                }
                "measure" => {
                    if parts.len() == 4 && parts[1] == "q" && parts[3] == "c" {
                        Ok(QASMInstruction::MeasureAll)
                    } else if parts.len() == 4 && parts[2] == "->" {
                        let (q_reg, q_index) = parse_qubit(parts[1])?;
                        let (c_reg, c_index) = parse_qubit(parts[3])?;
                        Ok(QASMInstruction::Measure(q_reg, q_index, c_reg, c_index))
                    } else {
                        Err(format!("Invalid measure format: {}", s))
                    }
                }
                "reset" => {
                    let (reg, index) = parse_qubit(parts[1])?;
                    Ok(QASMInstruction::Reset(reg, index))
                }
                _ => Err(format!("Unsupported instruction: {}", s)),
            }
        }
    }
}

pub fn parse_qasm(input: &str) -> Result<Vec<QASMInstruction>, String> {
    input
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("//"))
        .filter(|line| !line.starts_with("OPENQASM") && !line.starts_with("include"))
        .map(|line| line.trim_end_matches(';'))
        .map(QASMInstruction::from_str)
        .collect()
}

pub fn update_peak_memory() {
    if let Some(usage) = memory_stats() {
        let current_memory = usage.physical_mem;
        PEAK_MEMORY.fetch_max(current_memory, Ordering::Relaxed);
    }
}
