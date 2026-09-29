%% Lightweight verification of the active tensor utilities and dependencies.
clear; clc; rng default;

script_dir = fileparts(mfilename('fullpath'));
code_dir = fileparts(script_dir);
addpath(fullfile(code_dir, 'src', 'tensor'));
addpath(fullfile(code_dir, 'src', 'recovery'));

required_names = {'tt_tensor', 'my_chop2', 'fwht', 'ifwht'};
missing_names = required_names(cellfun(@(name) isempty(which(name)), ...
                                       required_names));
assert(isempty(missing_names), ...
    'Missing dependencies. Run setup_qtt_paths first: %s', ...
    strjoin(missing_names, ', '));

% Verify the general tensor contraction helper against an explicit product.
A = reshape(1:24, [2, 3, 4]);
B = reshape(1:20, [4, 5]);
actual = tns_mult(A, 3, B, 1);
expected = reshape(reshape(A, 6, 4) * B, [2, 3, 5]);
assert(norm(actual(:) - expected(:)) < 1e-12, ...
    'tns_mult failed the explicit contraction check.');

% Verify the Walsh--Hadamard transform convention used by the experiments.
x = randn(16, 1);
x_roundtrip = ifwht(fwht(x, 16, 'hadamard'), 16, 'hadamard');
assert(norm(x_roundtrip - x) / norm(x) < 1e-12, ...
    'FWHT/IFWHT round-trip check failed.');

% Verify the local TT-SVD implementation with a representable small tensor.
X = randn(2, 2, 2);
X_tt = my_ttsvd(X, [2, 2, 2], [1, 2, 2, 1]);
X_reconstructed = full(X_tt);
assert(norm(X_reconstructed(:) - X(:)) / norm(X(:)) < 1e-12, ...
    'my_ttsvd reconstruction check failed.');

fprintf('All smoke tests passed.\n');
