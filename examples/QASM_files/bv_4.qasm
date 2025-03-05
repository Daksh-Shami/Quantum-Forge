OPENQASM 3.0;
include "qelib1.inc";

// The circuit uses n+1 qubits, where n is the length of the hidden bitstring
// In this case, we're using a 3-bit hidden string, so we need 4 qubits total
qreg q[4];
creg c[3];

// Initialize the auxiliary qubit (q[3]) to |1>
x q[3];

// Apply Hadamard gates to all qubits
h q[0];
h q[1];
h q[2];
h q[3];

// Apply the oracle (assume the hidden string is 101)
cx q[0],q[3];
cx q[2],q[3];

// Apply Hadamard gates to the first 3 qubits
h q[0];
h q[1];
h q[2];

// Measure the first 3 qubits
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];