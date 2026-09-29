% This file aims to check the order of reshape function in matlab

% Check 1
A = 1:8
a = reshape(A, 2,2,2)



% Check 2
d = 3;
sz = 2 * ones(1,d);

X = randn(sz);
x = X(:);
