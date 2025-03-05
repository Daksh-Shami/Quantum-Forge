OPENQASM 2.0;
include "qelib1.inc";

// Define quantum registers
qreg q[5];

// Define classical registers
creg c[5];

// Apply Hadamard gate to the first qubit
h q[0];

// Apply Phase gate (S gate) to the second qubit
s q[1];

// Apply CNOT gate with control qubit 0 and target qubit 1
cx q[0],q[1];

// Apply Hadamard gate to the third qubit
h q[2];

// Apply Toffoli gate (CCX) with control qubits 0 and 1, and target qubit 2
ccx q[0],q[1],q[2];

// Apply CNOT gate with control qubit 2 and target qubit 3
cx q[2],q[3];

// Apply Hadamard gate to the fourth qubit
h q[3];

// Apply Phase gate (S gate) to the fourth qubit
s q[3];

// Apply SWAP gate between qubits 3 and 4
swap q[3],q[4];

// Apply CNOT gate with control qubit 4 and target qubit 0
cx q[4],q[0];

// Measure all qubits
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];