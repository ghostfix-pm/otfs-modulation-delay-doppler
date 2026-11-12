% OTFS modulation (inverse DFT along Doppler)

X_dt = M * ifft(X, M, 2); % inverse transform
s1 = reshape(X_dt, N * M, 1); 