N = 8;
x = randn(N);
A = dftmtx(N);
y1 = A*x;
y2 = fft(x);
norm(y1 - y2)
norm(ifft(y2) - x)
A;