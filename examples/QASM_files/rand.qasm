OPENQASM 2.0;
include "qelib1.inc";

// Quantum and classical registers
qreg q[12];
creg c[12];

// Initialize auxiliary qubit (q[11]) to |1>
x q[11];

// Apply Hadamard gates to all qubits
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];

// Apply the oracle (hidden string: 110110011010)
cx q[1], q[11];
cx q[2], q[11];
cx q[4], q[11];
cx q[5], q[11];
cx q[7], q[11];
cx q[9], q[11];

// Apply additional RZ rotations and phase gates
rz(pi/6) q[0];
s q[2];
rz(pi/8) q[4];
s q[6];
rz(pi/12) q[8];

// Apply CNOT chain for entanglement
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[4];
cx q[4], q[5];
cx q[5], q[6];
cx q[6], q[7];
cx q[7], q[8];
cx q[8], q[9];
cx q[9], q[10];
cx q[10], q[11];

// Apply Toffoli (CCX) gates for complexity
ccx q[0], q[1], q[2];
ccx q[3], q[4], q[5];
ccx q[6], q[7], q[8];
ccx q[9], q[10], q[11];

// Apply more RZ rotations for phase kickbacks
rz(pi/4) q[1];
rz(pi/5) q[3];
rz(pi/7) q[5];
rz(pi/9) q[7];
rz(pi/11) q[9];

// Final round of Hadamard gates
h q[0];
h q[2];
h q[4];
h q[6];
h q[8];
h q[10];

// Measurement
measure q -> c;
