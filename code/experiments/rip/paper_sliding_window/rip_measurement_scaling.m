%% ========================================================================
%  Empirical RIP experiment: vary the number of measurements
%
%  For each measurement budget, draw independent partial Walsh--Hadamard
%  operators. For each operator, draw an independent finite test set of
%  sliding-window tensors and record the largest observed relative energy
%  distortion. The plotted statistic is the mean of these operator-wise
%  maxima.
% ========================================================================
clear; clc; rng default;

% ---------------- experiment parameters ---------------------------------
d                = 12;          % tensor order
s                = 3;           % cluster order / window length
k                = 5;           % number of active components
p                = 2;           % step size between active windows
C_scale          = 1;           % scaling of active 2-vectors
num_operator_rep = 100;         % independent draws of A per M
num_test_tensors = 200;         % finite test set size per operator
M_list           = 25:25:750;   % measurement budgets
script_dir = fileparts(mfilename('fullpath'));
code_dir   = fullfile(script_dir, '..', '..', '..');
data_dir   = fullfile(code_dir, 'results', 'data', 'rip', 'paper_sliding_window');
figure_dir = fullfile(code_dir, 'results', 'figures', 'rip', 'paper_sliding_window');

assert(1 + (k-1)*p + s - 1 <= d, ...
       'Need a_k + s - 1 <= d for the sliding-window supports.');

% ---------------- precompute model structure -----------------------------
support_list = sliding_window_supports(k, s, p);
N = 2^d;

mean_empirical_delta = zeros(1, numel(M_list));
mean_tt_rank         = zeros(1, numel(M_list));
max_tt_rank          = zeros(1, numel(M_list));
all_empirical_delta  = cell(1, numel(M_list));
all_tt_rank          = cell(1, numel(M_list));
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

%% ========================================================================
%  Loop over measurement budgets
% ========================================================================
for m_idx = 1:numel(M_list)
    M = M_list(m_idx);
    fprintf('Current M: %d (%d of %d)\n', M, m_idx, numel(M_list));

    empirical_delta = zeros(1, num_operator_rep);
    tt_rank_mat = zeros(num_operator_rep, num_test_tensors);

    for op_idx = 1:num_operator_rep
        row_idx = randi(N, M, 1);
        max_distortion = 0;

        for test_idx = 1:num_test_tensors
            [g_qtt, max_rank] = sum_rank1_qtt( ...
                d, support_list, C_scale);
            g = full(g_qtt);
            g = g(:);

            g_hat = fwht(g, N, 'hadamard');
            y = (N/sqrt(M)) * g_hat(row_idx);
            distortion = abs(norm(y)^2 / norm(g)^2 - 1);

            max_distortion = max(max_distortion, distortion);
            tt_rank_mat(op_idx, test_idx) = max_rank;
        end

        empirical_delta(op_idx) = max_distortion;
    end

    mean_empirical_delta(m_idx) = mean(empirical_delta);
    mean_tt_rank(m_idx) = mean(tt_rank_mat(:));
    max_tt_rank(m_idx) = max(tt_rank_mat(:));
    all_empirical_delta{m_idx} = empirical_delta;
    all_tt_rank{m_idx} = tt_rank_mat;
end

%% ========================================================================
%  Plot: empirical RIP statistic vs. number of measurements
% ========================================================================
fig = figure; hold on; box on;

plot(M_list, mean_empirical_delta, 'o-', ...
     'LineWidth', 1.3, 'MarkerSize', 7);

reference_curve = mean_empirical_delta(1) * ...
                  (M_list(1)./M_list).^(1/2);
plot(M_list, reference_curve, '--k', 'LineWidth', 1.2);

set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('number of measurements M');
ylabel('empirical RIP distortion');
title(sprintf(['sliding window: d = %d, s = %d, k = %d, p = %d, ', ...
               'operators = %d, test set = %d'], ...
              d, s, k, p, num_operator_rep, num_test_tensors));
legend('mean over operators', 'M^{-1/2} scaling guide', ...
       'Location', 'best');
grid on;

%% ========================================================================
%  Save data and figure
% ========================================================================
experiment_name = sprintf( ...
    'rip_measurement_scaling_d%d_s%d_k%d_p%d_ops%d_tests%d', ...
    d, s, k, p, num_operator_rep, num_test_tensors);
data_path = fullfile(data_dir, [experiment_name, '.mat']);
fig_path  = fullfile(figure_dir, [experiment_name, '.fig']);
png_path  = fullfile(figure_dir, [experiment_name, '.png']);

save(data_path, 'd', 's', 'k', 'p', 'C_scale', ...
                'num_operator_rep', 'num_test_tensors', 'M_list', ...
                'mean_empirical_delta', 'all_empirical_delta', ...
                'mean_tt_rank', 'max_tt_rank', 'all_tt_rank', ...
                'reference_curve', 'support_list');
savefig(fig, fig_path);
exportgraphics(fig, png_path, 'Resolution', 300);

fprintf('Saved data to: %s\n', data_path);
fprintf('Saved MATLAB figure to: %s\n', fig_path);
fprintf('Saved PNG figure to: %s\n', png_path);

%% ========================================================================
%  Helper: sliding-window supports I_1,...,I_k of size s and step p
% ========================================================================
function support_list = sliding_window_supports(k, s, p)
    support_list = cell(1, k);
    for t = 1:k
        start_idx = 1 + (t-1)*p;
        support_list{t} = start_idx:(start_idx+s-1);
    end
end

%% ========================================================================
%  Helper: sum of rank-1 QTT tensors over a support list
% ========================================================================
function [g_qtt, max_rank] = sum_rank1_qtt(d, support_list, C)
    g_tensor = zeros(2*ones(1, d));
    for t = 1:numel(support_list)
        component = generate_qtt_rank1_tensor(d, support_list{t}, C);
        g_tensor = g_tensor + component;
    end

    g_qtt = tt_tensor(g_tensor);
    max_rank = max(g_qtt.r);
end

%% ========================================================================
%  Helper: rank-1 QTT tensor with active modes on a specified support
% ========================================================================
function g = generate_qtt_rank1_tensor(d, support, C)
    factors = cell(1, d);
    for j = 1:d
        factors{j} = [1; 0];
    end

    for j = support
        v = randn(2, 1);
        factors{j} = C * v / norm(v);
    end

    g = factors{d};
    for j = d-1:-1:1
        g = kron(g, factors{j});
    end
    g = reshape(g, 2*ones(1, d));
end
