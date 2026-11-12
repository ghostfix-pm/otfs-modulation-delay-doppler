function [X, GI] = generate_dd_frame(N, M, mod_size)

% Number of information symbols per frame:
N_syms_per_frame = N*M;

% Number of information bits per frame:
N_bits_per_frame = N_syms_per_frame * log2(mod_size);

% Random bit generation:
tx_info_bits = randi([0,1], N_bits_per_frame, 1);
% tx_info_bits = [0; 0; 0; 1; 1; 1; 1; 0; 0; 0; 0; 1; 1; 1; 1; 0];

% QAM modulation:
tx_info_symbols = qammod(tx_info_bits, mod_size, 'gray', 'InputType', 'bit');

% Build the N x M OTFS delay-Doppler frame and pilot grid

X = reshape(tx_info_symbols, N, M);

GI = zeros(N, M);
GI(N/2, M/2) = 10;

end