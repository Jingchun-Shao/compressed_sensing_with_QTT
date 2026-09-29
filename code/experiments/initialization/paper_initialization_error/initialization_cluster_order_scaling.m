%% ========================================================================
%  Initialization experiment: vary cluster order s
% ========================================================================
clear; clc; rng default;

% ---------------- experiment parameters ---------------------------------
d                = 12;          % tensor order
s_list           = 1:7;         % cluster orders / window lengths
k                = 4;           % number of active components
p                = 1;           % step size between active windows
M                = 800;         % fixed measurement budget
C_scale          = 1;           % scaling of active 2-vectors
r_trunc          = k;           % TT truncation rank used in initialization
num_operator_rep = 100;         % independent draws of A
num_test_tensors = 200;         % finite test set size per cluster order

script_dir = fileparts(mfilename('fullpath'));
code_dir   = fullfile(script_dir, '..', '..', '..');
addpath(fullfile(code_dir, 'src', 'tensor'));
data_dir   = fullfile(code_dir, 'results', 'data', 'initialization', 'paper_initialization_error');
figure_dir = fullfile(code_dir, 'results', 'figures', 'initialization', 'paper_initialization_error');

assert(all(1 + (k-1)*p + s_list - 1 <= d), ...
       'Need a_k + s - 1 <= d for every cluster order.');

if ~exist(data_dir, 'dir'), mkdir(data_dir); end
if ~exist(figure_dir, 'dir'), mkdir(figure_dir); end

N = 2^d;
mean_raw_bp_error = zeros(1, numel(s_list));
mean_bp_error   = zeros(1, numel(s_list));
mean_init_error = zeros(1, numel(s_list));
mean_adjoint_init_error = zeros(1, numel(s_list));
mean_tt_rank    = zeros(1, numel(s_list));
max_tt_rank     = zeros(1, numel(s_list));

all_raw_bp_error = cell(1, numel(s_list));
all_bp_error   = cell(1, numel(s_list));
all_init_error = cell(1, numel(s_list));
all_adjoint_init_error = cell(1, numel(s_list));
all_tt_rank    = cell(1, numel(s_list));
support_lists  = cell(1, numel(s_list));

%% ========================================================================
%  Loop over cluster orders
% ========================================================================
for s_idx = 1:numel(s_list)
    s = s_list(s_idx);
    fprintf('Current cluster order s: %d\n', s);

    support_list = sliding_window_supports(k, s, p);
    support_lists{s_idx} = support_list;

    raw_bp_error = zeros(1, num_operator_rep);
    bp_error = zeros(1, num_operator_rep);
    init_error = zeros(1, num_operator_rep);
    adjoint_init_error = zeros(1, num_operator_rep);
    tt_rank_mat = zeros(num_operator_rep, num_test_tensors);

    for op_idx = 1:num_operator_rep
        row_idx = randi(N, M, 1);
        max_raw_bp = 0;
        max_bp = 0;
        max_init = 0;
        max_adjoint_init = 0;

        for test_idx = 1:num_test_tensors
            % ---- generate one sliding-window structured tensor ----------
            [x_qtt, max_rank] = sum_rank1_qtt(d, support_list, C_scale);
            x = full(x_qtt);
            x = x(:);

            % ---- measurements and backprojection -----------------------
            x_hat = fwht(x, N, 'hadamard');
            y = (N / sqrt(M)) * x_hat(row_idx);
            z = accumarray(row_idx(:), y(:), [N, 1]);
            raw_bp = ifwht(z, N, 'hadamard') / sqrt(M);
            e_raw_bp = norm(raw_bp - x) / norm(x);

            % ---- low-order clustered projection P_{V_s} -----------------
            sz = 2 * ones(1, d);
            raw_tensor = reshape(raw_bp, sz);
            mask = build_cluster_mask(sz, s);
            bp_tensor = mask .* raw_tensor;
            bp = bp_tensor(:);

            e_bp = norm(bp - x) / norm(x);

            % ---- TT-SVD truncations ------------------------------------
            R = r_trunc * ones(d+1, 1);
            R(1) = 1;
            R(end) = 1;

            % Direct baseline: TT-SVD(A^*y), without P_{V_s} filtering.
            adjoint_init_tt = my_ttsvd(raw_bp, sz, R);
            adjoint_init_tensor = reshape(full(adjoint_init_tt), sz);
            e_adjoint_init = norm(adjoint_init_tensor(:) - x) / norm(x);

            % Existing proposed initialization: TT-SVD(P_{V_s}A^*y).
            init_tt = my_ttsvd(bp, sz, R);
            init_tensor = reshape(full(init_tt), sz);
            e_init = norm(init_tensor(:) - x) / norm(x);

            max_raw_bp = max(max_raw_bp, e_raw_bp);
            max_bp = max(max_bp, e_bp);
            max_init = max(max_init, e_init);
            max_adjoint_init = max(max_adjoint_init, e_adjoint_init);
            tt_rank_mat(op_idx, test_idx) = max_rank;
        end

        raw_bp_error(op_idx) = max_raw_bp;
        bp_error(op_idx) = max_bp;
        init_error(op_idx) = max_init;
        adjoint_init_error(op_idx) = max_adjoint_init;
    end

    mean_raw_bp_error(s_idx) = mean(raw_bp_error);
    mean_bp_error(s_idx)   = mean(bp_error);
    mean_init_error(s_idx) = mean(init_error);
    mean_adjoint_init_error(s_idx) = mean(adjoint_init_error);
    mean_tt_rank(s_idx)    = mean(tt_rank_mat(:));
    max_tt_rank(s_idx)     = max(tt_rank_mat(:));

    all_raw_bp_error{s_idx} = raw_bp_error;
    all_bp_error{s_idx}   = bp_error;
    all_init_error{s_idx} = init_error;
    all_adjoint_init_error{s_idx} = adjoint_init_error;
    all_tt_rank{s_idx}    = tt_rank_mat;
end

%% ========================================================================
%  Plot
% ========================================================================
fig = figure('Color', 'w', 'Units', 'inches', ...
             'Position', [1, 1, 5.2, 3.8]);
hold on; box on;

plot(s_list, mean_raw_bp_error, 'x-', 'LineWidth', 1.2, 'MarkerSize', 7);
plot(s_list, mean_bp_error, 'o-', 'LineWidth', 1.3, 'MarkerSize', 7);
plot(s_list, mean_init_error, '^-', 'LineWidth', 1.2, 'MarkerSize', 6);
plot(s_list, mean_adjoint_init_error, 's-', 'LineWidth', 1.2, 'MarkerSize', 6);

bp_theory_scale = zeros(1, numel(s_list));
init_theory_scale = zeros(1, numel(s_list));
delta_reference = 0.05;
for idx = 1:numel(s_list)
    D_s = clustered_dimension(d, s_list(idx));
    bp_theory_scale(idx) = sqrt(D_s * log(2*D_s/delta_reference) / M);
    init_theory_scale(idx) = (1 + sqrt(d - 1)) * bp_theory_scale(idx);
end
bp_reference_curve = mean_bp_error(1) * bp_theory_scale / bp_theory_scale(1);
init_reference_curve = mean_init_error(1) * init_theory_scale / init_theory_scale(1);
plot(s_list, bp_reference_curve, '--k', 'LineWidth', 1.2);
plot(s_list, init_reference_curve, ':k', 'LineWidth', 1.4);

set(gca, 'YScale', 'log');
xlabel('interaction order s');
ylabel('relative initialization error');
legend('Backprojection', 'After model projection', ...
       'Structure-aware initializer', 'Spectral initializer', ...
       'Projection guide', 'Initializer guide', ...
       'Location', 'best', 'FontSize', 9);
grid on;
set(gca, 'FontSize', 11, 'LineWidth', 0.9);

%% ========================================================================
%  Save data and figure
% ========================================================================
experiment_name = sprintf('init_cluster_order_scaling_d%d_k%d_p%d_M%d_ops%d_tests%d', ...
                          d, k, p, M, num_operator_rep, num_test_tensors);
data_path = fullfile(data_dir, [experiment_name, '.mat']);
fig_path  = fullfile(figure_dir, [experiment_name, '.fig']);
png_path  = fullfile(figure_dir, [experiment_name, '.png']);
pdf_path  = fullfile(figure_dir, [experiment_name, '.pdf']);

save(data_path, 'd', 's_list', 'k', 'p', 'M', 'C_scale', 'r_trunc', ...
                'num_operator_rep', 'num_test_tensors', 'delta_reference', ...
                'mean_raw_bp_error', 'mean_bp_error', ...
                'mean_init_error', 'mean_adjoint_init_error', ...
                'bp_reference_curve', 'init_reference_curve', ...
                'mean_tt_rank', 'max_tt_rank', ...
                'all_raw_bp_error', 'all_bp_error', 'all_init_error', ...
                'all_adjoint_init_error', 'all_tt_rank', ...
                'support_lists');
savefig(fig, fig_path);
exportgraphics(fig, png_path, 'Resolution', 300);
exportgraphics(fig, pdf_path, 'ContentType', 'vector');

fprintf('Saved data to: %s\n', data_path);
fprintf('Saved MATLAB figure to: %s\n', fig_path);
fprintf('Saved PNG figure to: %s\n', png_path);
fprintf('Saved PDF figure to: %s\n', pdf_path);

%% ========================================================================
%  Helpers
% ========================================================================
function support_list = sliding_window_supports(k, s, p)
    support_list = cell(1, k);
    for t = 1:k
        start_idx = 1 + (t-1)*p;
        support_list{t} = start_idx:(start_idx+s-1);
    end
end

function [g_qtt, max_rank] = sum_rank1_qtt(d, support_list, C)
    g_tensor = zeros(2*ones(1, d));
    for t = 1:numel(support_list)
        component = generate_qtt_rank1_tensor(d, support_list{t}, C);
        g_tensor = g_tensor + component;
    end

    g_qtt = tt_tensor(g_tensor);
    max_rank = max(g_qtt.r);
end

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

function mask = build_cluster_mask(sz, s)
    d = numel(sz);
    count_power_vec = 1;

    for i = 1:d
        vi = 2 * ones(sz(i), 1, 'double');
        vi(1) = 1;
        count_power_vec = kron(vi, count_power_vec);
    end

    count_order_vec = round(log(count_power_vec) / log(2));
    mask = reshape(count_order_vec <= s, sz);
end

function D_s = clustered_dimension(d, s)
    D_s = 0;
    for ell = 0:s
        D_s = D_s + nchoosek(d, ell);
    end
end
