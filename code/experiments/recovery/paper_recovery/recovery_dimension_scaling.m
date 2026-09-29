%% ========================================================================
%  Recovery experiment: vary tensor order d
%
%  Compares fixed-rank TT recovery from:
%    1. filtered backprojection + TT-SVD initialization;
%    2. direct adjoint backprojection TT-SVD(A^*y) initialization;
%    3. random TT initialization with the same TT-rank as the ground truth;
%    4. near-zero random tensor + TT-SVD initialization.
% ========================================================================
clear; clc; rng default;

% ---------------- experiment parameters ---------------------------------
d_list     = 9:16;            % tensor orders
s          = 3;               % cluster order / window length
k          = 3;               % number of active components
p          = 2;               % step size between active windows
M          = 400;             % fixed measurement budget
C_scale    = 1;               % scaling of active 2-vectors
num_rep    = 100;             % independent recovery trials per d
epoch      = 100;             % ALS sweeps
error_tol  = 1e-5;            % success threshold on final relative error
zero_init_rel_norm = 1e-6;    % near-zero seed norm relative to ground truth

script_dir = fileparts(mfilename('fullpath'));
code_dir   = fullfile(script_dir, '..', '..', '..');
addpath(fullfile(code_dir, 'src', 'tensor'));
addpath(fullfile(code_dir, 'src', 'recovery'));
data_dir   = fullfile(code_dir, 'results', 'data', 'recovery', 'paper_recovery');
figure_dir = fullfile(code_dir, 'results', 'figures', 'recovery', 'paper_recovery');

assert(all(1 + (k-1)*p + s - 1 <= d_list), ...
       'Need a_k + s - 1 <= d for every tensor order.');

if ~exist(data_dir, 'dir'), mkdir(data_dir); end
if ~exist(figure_dir, 'dir'), mkdir(figure_dir); end

success_rate_proxy  = zeros(1, numel(d_list));
success_rate_adjoint = zeros(1, numel(d_list));
success_rate_random = zeros(1, numel(d_list));
success_rate_zero   = zeros(1, numel(d_list));
mean_final_proxy    = zeros(1, numel(d_list));
mean_final_adjoint  = zeros(1, numel(d_list));
mean_final_random   = zeros(1, numel(d_list));
mean_final_zero     = zeros(1, numel(d_list));
median_final_proxy  = zeros(1, numel(d_list));
median_final_adjoint = zeros(1, numel(d_list));
median_final_random = zeros(1, numel(d_list));
median_final_zero   = zeros(1, numel(d_list));
mean_init_proxy     = zeros(1, numel(d_list));
mean_init_adjoint   = zeros(1, numel(d_list));
mean_init_random    = zeros(1, numel(d_list));
mean_init_zero      = zeros(1, numel(d_list));
mean_zero_norm_ratio = zeros(1, numel(d_list));
mean_iter_proxy     = zeros(1, numel(d_list));
mean_iter_adjoint   = zeros(1, numel(d_list));
mean_iter_random    = zeros(1, numel(d_list));
mean_iter_zero      = zeros(1, numel(d_list));
median_iter_proxy   = zeros(1, numel(d_list));
median_iter_adjoint = zeros(1, numel(d_list));
median_iter_random  = zeros(1, numel(d_list));
median_iter_zero    = zeros(1, numel(d_list));

all_final_proxy  = cell(1, numel(d_list));
all_final_adjoint = cell(1, numel(d_list));
all_final_random = cell(1, numel(d_list));
all_final_zero   = cell(1, numel(d_list));
all_init_proxy   = cell(1, numel(d_list));
all_init_adjoint = cell(1, numel(d_list));
all_init_random  = cell(1, numel(d_list));
all_init_zero    = cell(1, numel(d_list));
all_zero_norm_ratio = cell(1, numel(d_list));
all_curve_proxy  = cell(1, numel(d_list));
all_curve_adjoint = cell(1, numel(d_list));
all_curve_random = cell(1, numel(d_list));
all_curve_zero   = cell(1, numel(d_list));
all_iter_proxy   = cell(1, numel(d_list));
all_iter_adjoint = cell(1, numel(d_list));
all_iter_random  = cell(1, numel(d_list));
all_iter_zero    = cell(1, numel(d_list));
all_tt_rank      = cell(1, numel(d_list));
support_lists    = cell(1, numel(d_list));

experiment_name = sprintf('recovery_dimension_scaling_s%d_k%d_p%d_M%d_reps%d_epochs%d', ...
                          s, k, p, M, num_rep, epoch);
data_path = fullfile(data_dir, [experiment_name, '.mat']);

%% ========================================================================
%  Loop over tensor orders
% ========================================================================
for d_idx = 1:numel(d_list)
    d = d_list(d_idx);
    N = 2^d;
    sz = 2 * ones(1, d);
    fprintf('Current dimension d: %d\n', d);

    support_list = sliding_window_supports(k, s, p);
    support_lists{d_idx} = support_list;
    cluster_mask = build_cluster_mask(sz, s);

    final_proxy  = zeros(1, num_rep);
    final_adjoint = zeros(1, num_rep);
    final_random = zeros(1, num_rep);
    final_zero   = zeros(1, num_rep);
    init_proxy   = zeros(1, num_rep);
    init_adjoint = zeros(1, num_rep);
    init_random  = zeros(1, num_rep);
    init_zero    = zeros(1, num_rep);
    zero_norm_ratio = zeros(1, num_rep);
    iter_proxy   = zeros(1, num_rep);
    iter_adjoint = zeros(1, num_rep);
    iter_random  = zeros(1, num_rep);
    iter_zero    = zeros(1, num_rep);
    curve_proxy  = zeros(epoch, num_rep);
    curve_adjoint = zeros(epoch, num_rep);
    curve_random = zeros(epoch, num_rep);
    curve_zero   = zeros(epoch, num_rep);
    tt_rank_mat  = zeros(num_rep, d+1);

    for rep = 1:num_rep
        fprintf('  Trial %d/%d\n', rep, num_rep);

        [x_tensor, g_qtt] = sample_sliding_window_tensor(d, support_list, C_scale);
        x = x_tensor(:);
        R_true = g_qtt.r;
        tt_rank_mat(rep, :) = R_true(:).';

        row_idx = randi(N, M, 1);
        x_hat = fwht(x, N, 'hadamard');
        y = (N / sqrt(M)) * x_hat(row_idx);

        [proxy, adjoint_bp] = ...
            filtered_backprojection(row_idx, y, N, M, sz, cluster_mask);
        proxy_tt = my_ttsvd(proxy, sz, R_true);
        init_proxy(rep) = norm(full(proxy_tt) - x) / norm(x);

        adjoint_tt = my_ttsvd(adjoint_bp, sz, R_true);
        init_adjoint(rep) = norm(full(adjoint_tt) - x) / norm(x);

        random_tt = tt_random(sz, d, R_true);
        init_random(rep) = norm(full(random_tt) - x) / norm(x);

        zero_seed = randn(N, 1);
        zero_seed = zero_init_rel_norm * norm(x) * zero_seed / norm(zero_seed);
        zero_tt = my_ttsvd(zero_seed, sz, R_true);
        zero_full = full(zero_tt);
        zero_norm_ratio(rep) = norm(zero_full) / norm(x);
        init_zero(rep) = norm(zero_full - x) / norm(x);

        [~, err_proxy] = tt_iterative_opt_fwht(row_idx, y, proxy_tt, x, d, epoch);
        [~, err_adjoint] = tt_iterative_opt_fwht(row_idx, y, adjoint_tt, x, d, epoch);
        [~, err_random] = tt_iterative_opt_fwht(row_idx, y, random_tt, x, d, epoch);
        [~, err_zero] = tt_iterative_opt_fwht(row_idx, y, zero_tt, x, d, epoch);

        curve_proxy(:, rep) = err_proxy;
        curve_adjoint(:, rep) = err_adjoint;
        curve_random(:, rep) = err_random;
        curve_zero(:, rep) = err_zero;
        final_proxy(rep) = err_proxy(end);
        final_adjoint(rep) = err_adjoint(end);
        final_random(rep) = err_random(end);
        final_zero(rep) = err_zero(end);
        iter_proxy(rep) = first_success_epoch(err_proxy, error_tol, epoch);
        iter_adjoint(rep) = first_success_epoch(err_adjoint, error_tol, epoch);
        iter_random(rep) = first_success_epoch(err_random, error_tol, epoch);
        iter_zero(rep) = first_success_epoch(err_zero, error_tol, epoch);
    end

    success_rate_proxy(d_idx)  = mean(final_proxy < error_tol);
    success_rate_adjoint(d_idx) = mean(final_adjoint < error_tol);
    success_rate_random(d_idx) = mean(final_random < error_tol);
    success_rate_zero(d_idx)   = mean(final_zero < error_tol);
    mean_final_proxy(d_idx)    = mean(final_proxy);
    mean_final_adjoint(d_idx)  = mean(final_adjoint);
    mean_final_random(d_idx)   = mean(final_random);
    mean_final_zero(d_idx)     = mean(final_zero);
    median_final_proxy(d_idx)  = median(final_proxy);
    median_final_adjoint(d_idx) = median(final_adjoint);
    median_final_random(d_idx) = median(final_random);
    median_final_zero(d_idx)   = median(final_zero);
    mean_init_proxy(d_idx)     = mean(init_proxy);
    mean_init_adjoint(d_idx)   = mean(init_adjoint);
    mean_init_random(d_idx)    = mean(init_random);
    mean_init_zero(d_idx)      = mean(init_zero);
    mean_zero_norm_ratio(d_idx) = mean(zero_norm_ratio);
    mean_iter_proxy(d_idx)     = mean(iter_proxy);
    mean_iter_adjoint(d_idx)   = mean(iter_adjoint);
    mean_iter_random(d_idx)    = mean(iter_random);
    mean_iter_zero(d_idx)      = mean(iter_zero);
    median_iter_proxy(d_idx)   = median(iter_proxy);
    median_iter_adjoint(d_idx) = median(iter_adjoint);
    median_iter_random(d_idx)  = median(iter_random);
    median_iter_zero(d_idx)    = median(iter_zero);

    all_final_proxy{d_idx}  = final_proxy;
    all_final_adjoint{d_idx} = final_adjoint;
    all_final_random{d_idx} = final_random;
    all_final_zero{d_idx}   = final_zero;
    all_init_proxy{d_idx}   = init_proxy;
    all_init_adjoint{d_idx} = init_adjoint;
    all_init_random{d_idx}  = init_random;
    all_init_zero{d_idx}    = init_zero;
    all_zero_norm_ratio{d_idx} = zero_norm_ratio;
    all_curve_proxy{d_idx}  = curve_proxy;
    all_curve_adjoint{d_idx} = curve_adjoint;
    all_curve_random{d_idx} = curve_random;
    all_curve_zero{d_idx}   = curve_zero;
    all_iter_proxy{d_idx}   = iter_proxy;
    all_iter_adjoint{d_idx} = iter_adjoint;
    all_iter_random{d_idx}  = iter_random;
    all_iter_zero{d_idx}    = iter_zero;
    all_tt_rank{d_idx}      = tt_rank_mat;
end

%% ========================================================================
%  Plot
% ========================================================================
plot_recovery_panel(d_list, success_rate_proxy, success_rate_adjoint, ...
    success_rate_random, success_rate_zero, ...
    'tensor order d', 'success rate', ...
    sprintf('Recovery success, M = %d', M), [0, 1.05], false, [], ...
    fullfile(figure_dir, [experiment_name, '_success_rate']));

plot_recovery_panel(d_list, mean_final_proxy, mean_final_adjoint, ...
    mean_final_random, mean_final_zero, ...
    'tensor order d', 'mean final error', ...
    sprintf('Final error after %d sweeps', epoch), [], true, [], ...
    fullfile(figure_dir, [experiment_name, '_mean_final_error']));

plot_recovery_panel(d_list, mean_iter_proxy, mean_iter_adjoint, ...
    mean_iter_random, mean_iter_zero, ...
    'tensor order d', 'mean sweeps to tolerance', ...
    sprintf('Convergence speed, tol = %.0e', error_tol), ...
    [], false, epoch + 1, ...
    fullfile(figure_dir, [experiment_name, '_mean_iterations']));

%% ========================================================================
%  Save data
% ========================================================================
save(data_path, 'd_list', 's', 'k', 'p', 'M', 'C_scale', ...
     'num_rep', 'epoch', 'error_tol', 'zero_init_rel_norm', ...
     'support_lists', ...
     'success_rate_proxy', 'success_rate_adjoint', ...
     'success_rate_random', 'success_rate_zero', ...
     'mean_final_proxy', 'mean_final_adjoint', ...
     'mean_final_random', 'mean_final_zero', ...
     'median_final_proxy', 'median_final_adjoint', ...
     'median_final_random', 'median_final_zero', ...
     'mean_init_proxy', 'mean_init_adjoint', ...
     'mean_init_random', 'mean_init_zero', ...
     'mean_zero_norm_ratio', ...
     'mean_iter_proxy', 'mean_iter_adjoint', ...
     'mean_iter_random', 'mean_iter_zero', ...
     'median_iter_proxy', 'median_iter_adjoint', ...
     'median_iter_random', 'median_iter_zero', ...
     'all_final_proxy', 'all_final_adjoint', ...
     'all_final_random', 'all_final_zero', ...
     'all_init_proxy', 'all_init_adjoint', ...
     'all_init_random', 'all_init_zero', ...
     'all_zero_norm_ratio', ...
     'all_curve_proxy', 'all_curve_adjoint', ...
     'all_curve_random', 'all_curve_zero', ...
     'all_iter_proxy', 'all_iter_adjoint', ...
     'all_iter_random', 'all_iter_zero', ...
     'all_tt_rank');

fprintf('Saved data to: %s\n', data_path);

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

function [x_tensor, x_qtt] = sample_sliding_window_tensor(d, support_list, C_scale)
    x_tensor = zeros(2*ones(1, d));

    for t = 1:numel(support_list)
        factors = cell(1, d);
        for j = 1:d
            factors{j} = [1; 0];
        end

        for j = support_list{t}
            v = randn(2, 1);
            factors{j} = C_scale * v / norm(v);
        end

        component = factors{d};
        for j = d-1:-1:1
            component = kron(component, factors{j});
        end
        x_tensor = x_tensor + reshape(component, 2*ones(1, d));
    end

    x_qtt = tt_tensor(x_tensor);
end

function [proxy, raw_bp] = filtered_backprojection(row_idx, y, N, M, sz, cluster_mask)
    z = accumarray(row_idx(:), y(:), [N, 1]);
    raw_bp = ifwht(z, N, 'hadamard') / sqrt(M);
    raw_tensor = reshape(raw_bp, sz);
    proxy_tensor = cluster_mask .* raw_tensor;
    proxy = proxy_tensor(:);
end

function iter = first_success_epoch(error_curve, error_tol, epoch)
    iter = find(error_curve < error_tol, 1, 'first');
    if isempty(iter)
        iter = epoch + 1;
    end
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

function plot_recovery_panel(x, y_proxy, y_adjoint, y_random, y_zero, ...
                             x_label, y_label, ~, y_limits, ...
                             use_log_y, cap_line, out_path)
    fig = figure('Color', 'w', 'Units', 'inches', ...
                 'Position', [1, 1, 5.2, 3.8]);
    ax = axes(fig);
    hold(ax, 'on');
    box(ax, 'on');

    proxy_color = [0.0000, 0.4470, 0.7410];
    adjoint_color = [0.4660, 0.6740, 0.1880];
    random_color = [0.8500, 0.3250, 0.0980];
    zero_color = [0.4940, 0.1840, 0.5560];

    if use_log_y
        set(ax, 'YScale', 'log');
        semilogy(ax, x, y_proxy, 'o-', 'Color', proxy_color, ...
            'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', 'w');
        semilogy(ax, x, y_adjoint, '^-', 'Color', adjoint_color, ...
            'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', 'w');
        semilogy(ax, x, y_random, 's-', 'Color', random_color, ...
            'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', 'w');
        semilogy(ax, x, y_zero, 'd-', 'Color', zero_color, ...
            'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', 'w');
    else
        plot(ax, x, y_proxy, 'o-', 'Color', proxy_color, ...
            'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', 'w');
        plot(ax, x, y_adjoint, '^-', 'Color', adjoint_color, ...
            'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', 'w');
        plot(ax, x, y_random, 's-', 'Color', random_color, ...
            'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', 'w');
        plot(ax, x, y_zero, 'd-', 'Color', zero_color, ...
            'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', 'w');
    end

    if ~isempty(cap_line)
        yline(ax, cap_line, ':', 'Color', [0.25, 0.25, 0.25], ...
              'LineWidth', 1.2);
    end

    xlabel(ax, x_label, 'FontSize', 16);
    ylabel(ax, y_label, 'FontSize', 16);
    grid(ax, 'on');
    ax.GridAlpha = 0.18;
    ax.LineWidth = 0.9;
    ax.FontSize = 14;

    if ~isempty(y_limits)
        ylim(ax, y_limits);
    end

    savefig(fig, [out_path, '.fig']);
    exportgraphics(fig, [out_path, '.png'], 'Resolution', 300);
    exportgraphics(fig, [out_path, '.pdf'], 'ContentType', 'vector');
    close(fig);
end
