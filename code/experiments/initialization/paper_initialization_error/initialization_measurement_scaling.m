%% ========================================================================
%  Initialization experiment: vary number of Walsh--Hadamard measurements
%
%  For each M, draw several independent partial Walsh--Hadamard operators.
%  For each operator, draw an independent finite sliding-window test set
%  and record the largest observed initialization error over that test set.
% ========================================================================
clear; clc; rng default;

% ---------------- experiment parameters ---------------------------------
d                = 10;          % tensor order
s                = 3;           % cluster order / window length
k                = 3;           % number of active components
p                = 2;           % step size between active windows
C_scale          = 1;           % scaling of active 2-vectors
r_trunc          = k;           % TT truncation rank used in initialization
num_operator_rep = 100;         % independent draws of A
num_test_tensors = 200;         % finite test set size per operator
M_list           = 150:25:800;  % measurement budgets

script_dir = fileparts(mfilename('fullpath'));
code_dir   = fullfile(script_dir, '..', '..', '..');
addpath(fullfile(code_dir, 'src', 'tensor'));
data_dir   = fullfile(code_dir, 'results', 'data', 'initialization', 'paper_initialization_error');
figure_dir = fullfile(code_dir, 'results', 'figures', 'initialization', 'paper_initialization_error');

assert(1 + (k-1)*p + s - 1 <= d, ...
       'Need a_k + s - 1 <= d for the sliding-window supports.');

if ~exist(data_dir, 'dir'), mkdir(data_dir); end
if ~exist(figure_dir, 'dir'), mkdir(figure_dir); end

N = 2^d;
support_list = sliding_window_supports(k, s, p);

mean_raw_bp_error = zeros(1, numel(M_list));
mean_bp_error   = zeros(1, numel(M_list));
mean_init_error = zeros(1, numel(M_list));
mean_adjoint_init_error = zeros(1, numel(M_list));
mean_tt_rank    = zeros(1, numel(M_list));
max_tt_rank     = zeros(1, numel(M_list));

all_raw_bp_error = cell(1, numel(M_list));
all_bp_error   = cell(1, numel(M_list));
all_init_error = cell(1, numel(M_list));
all_adjoint_init_error = cell(1, numel(M_list));
all_tt_rank    = cell(1, numel(M_list));

%% ========================================================================
%  Loop over measurement budgets
% ========================================================================
for m_idx = 1:numel(M_list)
    M = M_list(m_idx);
    fprintf('Current M: %d\n', M);

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
            % Todo: Think which r makes more sense, shall we also truncate
            % by the true tt-rank
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

    mean_raw_bp_error(m_idx) = mean(raw_bp_error);
    mean_bp_error(m_idx)   = mean(bp_error);
    mean_init_error(m_idx) = mean(init_error);
    mean_adjoint_init_error(m_idx) = mean(adjoint_init_error);
    mean_tt_rank(m_idx)    = mean(tt_rank_mat(:));
    max_tt_rank(m_idx)     = max(tt_rank_mat(:));

    all_raw_bp_error{m_idx} = raw_bp_error;
    all_bp_error{m_idx}   = bp_error;
    all_init_error{m_idx} = init_error;
    all_adjoint_init_error{m_idx} = adjoint_init_error;
    all_tt_rank{m_idx}    = tt_rank_mat;
end

%% ========================================================================
%  Plot
% ========================================================================
fig = figure('Color', 'w', 'Units', 'inches', ...
             'Position', [1, 1, 5.2, 3.8]);
hold on; box on;

plot(M_list, mean_raw_bp_error, 'x-', 'LineWidth', 1.2, 'MarkerSize', 7);
plot(M_list, mean_bp_error, 'o-', 'LineWidth', 1.3, 'MarkerSize', 7);
plot(M_list, mean_init_error, '^-', 'LineWidth', 1.2, 'MarkerSize', 6);
plot(M_list, mean_adjoint_init_error, 's-', 'LineWidth', 1.2, 'MarkerSize', 6);

bp_reference_curve = mean_bp_error(1) * (M_list(1)./M_list).^(1/2);
init_reference_curve = mean_init_error(1) * (M_list(1)./M_list).^(1/2);
plot(M_list, bp_reference_curve, '--k', 'LineWidth', 1.2);
plot(M_list, init_reference_curve, ':k', 'LineWidth', 1.4);

set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('number of measurements m');
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
experiment_name = sprintf('init_measurement_scaling_d%d_s%d_k%d_p%d_ops%d_tests%d', ...
                          d, s, k, p, num_operator_rep, num_test_tensors);
data_path = fullfile(data_dir, [experiment_name, '.mat']);
fig_path  = fullfile(figure_dir, [experiment_name, '.fig']);
png_path  = fullfile(figure_dir, [experiment_name, '.png']);
pdf_path  = fullfile(figure_dir, [experiment_name, '.pdf']);

save(data_path, 'd', 's', 'k', 'p', 'C_scale', 'r_trunc', ...
                'num_operator_rep', 'num_test_tensors', 'M_list', ...
                'mean_raw_bp_error', 'mean_bp_error', ...
                'mean_init_error', 'mean_adjoint_init_error', ...
                'bp_reference_curve', 'init_reference_curve', ...
                'mean_tt_rank', 'max_tt_rank', ...
                'all_raw_bp_error', 'all_bp_error', 'all_init_error', ...
                'all_adjoint_init_error', 'all_tt_rank', ...
                'support_list');
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
