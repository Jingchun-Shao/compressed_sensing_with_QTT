function [x_qtt, error_list] = tt_iterative_opt_fft(row_idx, y, x_qtt, x_benchmark, d1, epoch)
%TT_ITERATIVE_OPT_FFT Alternating TT recovery with implicit partial FFT rows.
%
%   This fixed-rank optimizer has the same ALS structure as
%   tt_iterative_opt, but it avoids forming the dense Fourier matrix. For
%   each TT core, it builds the local design matrix by applying FFTs to the
%   basis tensors induced by the current left and right TT environments.

    row_idx = row_idx(:);
    y = y(:);
    M = numel(row_idx);
    N = prod(double(x_qtt.n));

    if numel(x_benchmark) ~= N
        error('x_benchmark must have prod(x_qtt.n) entries.');
    end
    if max(row_idx) > N || min(row_idx) < 1
        error('row_idx entries must be between 1 and prod(x_qtt.n).');
    end
    if numel(y) ~= M
        error('y must have the same number of entries as row_idx.');
    end

    error_list = zeros(epoch, 1);

    fprintf('Before optimization: %.6e\n', ...
            norm(full(x_qtt) - x_benchmark) / norm(x_benchmark));

    for ep = 1:epoch
        for core_idx = 1:d1
            A_local = local_partial_fft_design(row_idx, x_qtt, core_idx, M, N);
            core_update = solve_local_least_squares(A_local, y);

            ps = x_qtt.ps;
            x_qtt.core(ps(core_idx):ps(core_idx+1)-1) = core_update;
        end

        error_list(ep) = norm(full(x_qtt) - x_benchmark) / norm(x_benchmark);
    end

    fprintf('After optimization: %.6e\n', error_list(end));
end

function x = solve_local_least_squares(A, b)
    if exist('lsqminnorm', 'file') == 2
        x = lsqminnorm(A, b);
    else
        x = pinv(A) * b;
    end
end

function A_local = local_partial_fft_design(row_idx, x_qtt, core_idx, M, N)
    n = double(x_qtt.n(:).');
    r = double(x_qtt.r(:).');
    ni = n(core_idx);
    ri = r(core_idx);
    rnext = r(core_idx + 1);
    left_dim = prod(n(1:core_idx-1));
    right_dim = prod(n(core_idx+1:end));
    num_core_params = ri * ni * rnext;

    left_env = left_environment(x_qtt, core_idx, left_dim, ri);
    right_env = right_environment(x_qtt, core_idx, right_dim, rnext);
    basis_vectors = zeros(N, num_core_params);

    for param_idx = 1:num_core_params
        [left_rank_idx, mode_idx, right_rank_idx] = ind2sub([ri, ni, rnext], param_idx);
        mode_basis = zeros(ni, 1);
        mode_basis(mode_idx) = 1;

        basis_vectors(:, param_idx) = kron(right_env(right_rank_idx, :).', ...
                                           kron(mode_basis, left_env(:, left_rank_idx)));
    end

    fft_basis = fft(basis_vectors, [], 1);
    A_local = fft_basis(row_idx, :) / sqrt(M);
end

function left_env = left_environment(x_qtt, core_idx, left_dim, ri)
    if core_idx == 1
        left_env = 1;
        return;
    end

    left_tensor = tt_contraction(x_qtt, 1, core_idx-1, false, true);
    left_env = reshape(left_tensor, [left_dim, ri]);
end

function right_env = right_environment(x_qtt, core_idx, right_dim, rnext)
    if core_idx == x_qtt.d
        right_env = 1;
        return;
    end

    right_tensor = tt_contraction(x_qtt, core_idx+1, x_qtt.d, true, false);
    right_env = reshape(right_tensor, [rnext, right_dim]);
end
