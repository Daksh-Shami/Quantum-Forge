OPENQASM 3.0;

// Define quantum registers
qubit[5] q;

// Define classical registers
bit[5] c;

// Apply Hadamard gate to the first qubit
h q[0];

// Apply Phase gate (S gate) to the second qubit
s q[1];

// Apply CNOT gate with control qubit 0 and target qubit 1
cx q[0], q[1];

// Apply Toffoli gate (CCX) with control qubits 0 and 1, and target qubit 2
ccx q[0], q[1], q[2];

// Apply SWAP gate between qubits 3 and 4
swap q[3], q[4];

// Measure all qubits
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
c[4] = measure q[4];