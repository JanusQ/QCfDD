OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c0[12];
x q[1];
rz(-1.570796327179587) q[1];
x q[2];
rz(3.1415926499999998) q[2];
cx q[2],q[1];
x q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[2],q[1];
rz(-1.5707963271795862) q[5];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[5],q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
rz(-1.546360506310407e-08) q[8];
sx q[8];
rz(-3.1415926499999998) q[8];
sx q[8];
rz(1.546360550719328e-08) q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[5],q[3];
cx q[3],q[5];
cx q[3],q[2];
cx q[2],q[3];
cx q[2],q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[3];
sx q[3];
sx q[5];
rz(-pi/2) q[5];
rz(-3.1390812211185226) q[8];
sx q[8];
x q[9];
rz(1.5707963300000003) q[9];
sx q[10];
sx q[10];
rz(-1.570796308126189) q[10];
rz(1.5707963299999994) q[11];
cx q[11],q[8];
sx q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[8];
cx q[11],q[8];
rz(-pi/2) q[11];
sx q[11];
rz(1.5682848943236252) q[11];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[5],q[8];
sx q[5];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[8];
cx q[5],q[8];
rz(-pi) q[5];
sx q[5];
rz(-pi) q[5];
cx q[5],q[3];
rz(pi/2) q[3];
sx q[5];
rz(-pi/2) q[5];
sx q[5];
cx q[5],q[3];
rz(pi/2) q[3];
sx q[3];
cx q[3],q[2];
rz(pi/2) q[2];
sx q[3];
rz(-pi/2) q[3];
sx q[3];
cx q[3],q[2];
rz(pi/2) q[2];
sx q[2];
rz(-pi) q[2];
cx q[1],q[2];
sx q[3];
rz(pi/2) q[3];
sx q[8];
rz(pi/2) q[8];
x q[13];
x q[14];
cx q[11],q[14];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[11],q[14];
cx q[8],q[9];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[8],q[9];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
sx q[11];
rz(-pi/2) q[11];
x q[8];
rz(-pi/2) q[8];
cx q[5],q[8];
sx q[5];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[8];
cx q[5],q[8];
rz(-pi) q[5];
sx q[5];
rz(-pi) q[5];
cx q[5],q[3];
rz(pi/2) q[3];
sx q[5];
rz(-pi/2) q[5];
sx q[5];
cx q[5],q[3];
rz(pi/2) q[3];
sx q[3];
rz(-pi) q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
rz(-pi/2) q[5];
sx q[5];
rz(-pi) q[5];
sx q[8];
cx q[8],q[9];
sx q[8];
x q[16];
rz(3.1415926499999998) q[16];
cx q[14],q[16];
cx q[13],q[14];
cx q[14],q[13];
sx q[13];
cx q[12],q[13];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[13];
cx q[12],q[13];
rz(-pi/2) q[12];
sx q[12];
rz(-pi/2) q[12];
cx q[12],q[10];
rz(pi/2) q[13];
sx q[13];
rz(-pi) q[13];
x q[14];
rz(-pi/2) q[14];
cx q[11],q[14];
sx q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[14];
cx q[11],q[14];
rz(-pi) q[11];
sx q[11];
rz(-pi) q[11];
cx q[11],q[8];
sx q[11];
rz(-pi/2) q[11];
sx q[11];
sx q[14];
cx q[14],q[16];
sx q[14];
cx q[13],q[14];
sx q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[14];
cx q[13],q[14];
rz(-pi/2) q[13];
sx q[13];
rz(-pi) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[13],q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(pi/2) q[14];
sx q[14];
rz(-pi) q[14];
rz(pi/2) q[8];
cx q[11],q[8];
rz(-pi/2) q[11];
sx q[11];
rz(-pi) q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[16];
cx q[14],q[11];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
sx q[13];
sx q[13];
rz(-3.1415926381261885) q[13];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
rz(pi/2) q[8];
sx q[8];
rz(-pi) q[8];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[9];
cx q[8],q[5];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[5],q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[13];
sx q[14];
sx q[14];
rz(-3.1415926381261885) q[14];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[10];
cx q[10],q[12];
x q[14];
rz(-pi/2) q[14];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[5],q[3];
cx q[3],q[5];
cx q[3],q[2];
cx q[2],q[3];
cx q[2],q[1];
rz(-pi/2) q[1];
sx q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[3];
sx q[3];
sx q[5];
rz(-pi/2) q[5];
cx q[8],q[9];
cx q[11],q[8];
cx q[8],q[11];
rz(-pi) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[14];
sx q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[14];
cx q[11],q[14];
sx q[11];
rz(-pi) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
x q[8];
rz(-pi/2) q[8];
cx q[5],q[8];
sx q[5];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[8];
cx q[5],q[8];
rz(-pi) q[5];
sx q[5];
rz(-pi) q[5];
cx q[5],q[3];
rz(pi/2) q[3];
sx q[5];
rz(-pi/2) q[5];
sx q[5];
cx q[5],q[3];
rz(pi/2) q[3];
sx q[3];
cx q[3],q[2];
rz(pi/2) q[2];
sx q[3];
rz(-pi/2) q[3];
sx q[3];
cx q[3],q[2];
rz(pi/2) q[2];
sx q[2];
cx q[2],q[1];
rz(pi/2) q[1];
sx q[2];
rz(-pi/2) q[2];
sx q[2];
cx q[2],q[1];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi) q[2];
rz(-pi/2) q[3];
sx q[3];
rz(-pi) q[3];
rz(-pi/2) q[5];
sx q[5];
rz(-pi) q[5];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[9];
sx q[8];
rz(-pi) q[8];
cx q[11],q[8];
sx q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[8];
cx q[11],q[8];
rz(pi/2) q[11];
sx q[11];
rz(-pi/2) q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[16],q[14];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[11],q[14];
cx q[12],q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[11],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[10];
cx q[10],q[12];
rz(-pi/2) q[10];
sx q[10];
rz(-pi/2) q[8];
sx q[8];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[9];
cx q[8],q[5];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[16],q[14];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(-pi) q[12];
cx q[12],q[10];
rz(pi/2) q[10];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
cx q[12],q[10];
rz(pi/2) q[10];
sx q[10];
rz(-pi) q[10];
cx q[16],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[8],q[9];
cx q[8],q[5];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[5],q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
sx q[5];
sx q[5];
rz(1.5707963299999994) q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[11];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[11],q[8];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[3],q[2];
cx q[2],q[3];
cx q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(1.5707963235897928) q[1];
sx q[1];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
cx q[5],q[8];
cx q[9],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
x q[13];
rz(-pi/2) q[13];
cx q[12],q[13];
sx q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[13];
cx q[12],q[13];
rz(-pi) q[12];
sx q[12];
cx q[10],q[12];
rz(pi/2) q[13];
sx q[13];
rz(-pi/2) q[14];
sx q[14];
cx q[5],q[8];
cx q[3],q[5];
cx q[5],q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[9],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
rz(-pi) q[11];
cx q[11],q[14];
sx q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[14];
cx q[11],q[14];
rz(pi/2) q[14];
sx q[14];
rz(-pi) q[14];
cx q[16],q[14];
x q[14];
rz(-pi/2) q[14];
cx q[13],q[14];
sx q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[14];
cx q[13],q[14];
rz(-pi) q[13];
sx q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[10],q[12];
sx q[14];
rz(pi/2) q[14];
cx q[5],q[8];
cx q[8],q[5];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[9],q[8];
x q[8];
rz(-pi/2) q[8];
cx q[11],q[8];
sx q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[8];
cx q[11],q[8];
rz(-pi) q[11];
sx q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[16],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
sx q[8];
rz(pi/2) q[8];
cx q[5],q[8];
cx q[8],q[5];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[5];
cx q[9],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[16],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[13];
cx q[5],q[8];
cx q[8],q[5];
cx q[9],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[3],q[5];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[5],q[8];
cx q[8],q[5];
cx q[5],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[16],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi) q[16];
sx q[16];
rz(2.7488935735897932) q[16];
sx q[16];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[5],q[8];
cx q[8],q[11];
cx q[11],q[8];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[10],q[12];
rz(-pi) q[12];
sx q[12];
rz(1.5707963235897928) q[12];
sx q[12];
sx q[9];
rz(1.570796326410207) q[9];
sx q[9];
rz(-pi) q[9];
barrier q[2],q[1],q[9],q[3],q[16],q[5],q[11],q[8],q[13],q[14],q[10],q[12];
measure q[2] -> c0[0];
measure q[1] -> c0[1];
measure q[9] -> c0[2];
measure q[3] -> c0[3];
measure q[16] -> c0[4];
measure q[5] -> c0[5];
measure q[11] -> c0[6];
measure q[8] -> c0[7];
measure q[13] -> c0[8];
measure q[14] -> c0[9];
measure q[10] -> c0[10];
measure q[12] -> c0[11];
