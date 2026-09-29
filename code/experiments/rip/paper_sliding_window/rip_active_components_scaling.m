%% ========================================================================
%  Empirical RIP experiment: vary number of active components k
%
%  For each k, draw several independent partial Walsh--Hadamard operators.
%  For each operator, draw an independent finite test set of sliding-window
%  clustered QTT tensors and approximate the model-class supremum by its
%  maximum observed distortion.
% ========================================================================
clear; clc; rng default;

% ---------------- experiment parameters ---------------------------------
d                = 16;          % tensor order
s                = 3;           % cluster order / window length
k_list           = 1:7;         % number of active components
p                = 2;           % step size between active windows
M                = 200;         % fixed measurement budget
C_scale          = 1;           % scaling of active 2-vectors
num_operator_rep = 100;         % independent draws of A
num_test_tensors = 200;          % finite test set size per k
script_dir = fileparts(mfilename('fullpath'));
code_dir   = fullfile(script_dir, '..', '..', '..');
data_dir   = fullfile(code_dir, 'results', 'data', 'rip', 'paper_sliding_window');
figure_dir = fullfile(code_dir, 'results', 'figures', 'rip', 'paper_sliding_window');

assert(all(1 + (k_list-1)*p + s - 1 <= d), ...
       'Need a_k + s - 1 <= d for every active-component count.');

if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

% ---------------- pre-allocate storage ----------------------------------
N = 2^d;
mean_empirical_delta = zeros(1, numel(k_list));
mean_tt_rank         = zeros(1, numel(k_list));
max_tt_rank          = zeros(1, numel(k_list));

all_empirical_delta = cell(1, numel(k_list));
all_tt_rank         = cell(1, numel(k_list));
support_lists       = cell(1, numel(k_list));

%% ========================================================================
%  Loop over active-component counts
% ========================================================================
for k_idx = 1:numel(k_list)
    k = k_list(k_idx);
    fprintf('Current number of active components k: %d\n', k);

    support_list = sliding_window_supports(k, s, p);
    support_lists{k_idx} = support_list;

    empirical_delta = zeros(1, num_operator_rep);
    tt_rank_mat = zeros(num_operator_rep, num_test_tensors);

    for op_idx = 1:num_operator_rep
        row_idx = randi(N, M, 1);
        max_distortion = 0;

        for test_idx = 1:num_test_tensors
            [g_qtt, max_rank] = sum_rank1_qtt(d, support_list, C_scale);
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

    mean_empirical_delta(k_idx) = mean(empirical_delta);
    mean_tt_rank(k_idx)         = mean(tt_rank_mat(:));
    max_tt_rank(k_idx)          = max(tt_rank_mat(:));

    all_empirical_delta{k_idx} = empirical_delta;
    all_tt_rank{k_idx} = tt_rank_mat;
end

%% ========================================================================
%  Plot: empirical RIP distortion vs. active components
% ========================================================================
fig = figure; hold on; box on;

plot(k_list, mean_empirical_delta, 'o-', 'LineWidth', 1.3, 'MarkerSize', 7);

ylim([0.2, 0.4]);
xlabel('number of active components k');
ylabel('empirical RIP distortion');
title(sprintf('sliding window: d = %d, s = %d, p = %d, M = %d, test set = %d', ...
              d, s, p, M, num_test_tensors));
legend('mean over operators', 'Location', 'best');
grid on;

fprintf('Mean TT-rank over active-component counts:\n');
disp(mean_tt_rank);
fprintf('Maximum TT-rank over active-component counts:\n');
disp(max_tt_rank);

%% ========================================================================
%  Save data and figure
% ========================================================================
experiment_name = sprintf('rip_active_components_scaling_d%d_s%d_p%d_M%d_ops%d_tests%d', ...
                          d, s, p, M, num_operator_rep, num_test_tensors);
data_path = fullfile(data_dir, [experiment_name, '.mat']);
fig_path  = fullfile(figure_dir, [experiment_name, '.fig']);
png_path  = fullfile(figure_dir, [experiment_name, '.png']);

save(data_path, 'd', 's', 'k_list', 'p', 'M', 'C_scale', ...
                'num_operator_rep', 'num_test_tensors', ...
                'mean_empirical_delta', ...
                'mean_tt_rank', 'max_tt_rank', ...
                'all_empirical_delta', 'all_tt_rank', ...
                'support_lists');
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
