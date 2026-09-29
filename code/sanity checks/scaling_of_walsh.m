%% Verify Walsh--Hadamard ordering and scaling for QTT vectorization
clear; clc; rng default;

d = 4;
sz = 2 * ones(1, d);
N = 2^d;
H2 = [1, 1; 1, -1];


%% 1. Build H_2^{\otimes d} in MATLAB vectorization order
% For a d-way tensor, MATLAB uses
% vec(X x_1 H2 x_2 ... x_d H2) = (H2_d \otimes ... \otimes H2_1) vec(X).
H_kron = 1;
for mode = d:-1:1
    H_kron = kron(H_kron, H2);
end


X = randn(sz);
x = X(:);

y = fwht(x, N, "hadamard");
y_bench = (H_kron / N) * x;

norm(y - y_bench)


% Check 2: in matrix format
hadamard(N);
fwht(hadamard(N), N, 'hadamard')



