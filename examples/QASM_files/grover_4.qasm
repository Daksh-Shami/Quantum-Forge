OPENQASM 3.0;

qubit[4] q;
bit[4] c;

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

c = measure q;
