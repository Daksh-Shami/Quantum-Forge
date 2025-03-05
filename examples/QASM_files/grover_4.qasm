OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg c[4];

h q[0];
h q[1];
h q[2];
h q[3];

x q[0];
x q[2];
h q[3];
ccx q[0], q[1], q[2];
ccx q[2], q[3], q[1];
ccx q[0], q[1], q[2];
ccx q[2], q[3], q[1];
h q[3];
x q[0];
x q[2];

h q[0];
h q[1];
h q[2];
h q[3];
x q[0];
x q[1];
x q[2];
x q[3];
h q[3];
ccx q[0], q[1], q[2];
ccx q[2], q[3], q[1];
ccx q[0], q[1], q[2];
ccx q[2], q[3], q[1];
h q[3];
x q[0];
x q[1];
x q[2];
x q[3];
h q[0];
h q[1];
h q[2];
h q[3];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];