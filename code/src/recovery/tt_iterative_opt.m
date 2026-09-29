%%%% Recover the groundtruth tensor train by iteratively update the cores.
%%%% Updated version: fix the bug with tt_rank=1

function [x_qtt, error_list] = tt_iterative_opt(F_subsampled, y, x_qtt, x_benchmark, N, d1, epoch)
%   Optimizes the TT cores iteratively 
%
%   Inputs:
%       F_subsampled - Subsampled Fourier matrix, size (N, 2^d1)
%       y            - Observation vector.
%       x_qtt        - Initial guess in TT format, dimension d1
%       x_benchmark  - Ground truth benchmark vector.
%       N            - row size of the measurement matrix
%       d1           - dimension of x_qtt
%       epoch        - Each core is optimized once per epoch
%       
%
%   Outputs:
%       x_qtt        - Optimized TT vector.
%       error_list   - List of errors at each epoch.

    % Initialize error list
    error_list = zeros(epoch, 1);

    disp('Before optimization:');
    disp(norm(full(x_qtt) - x_benchmark));

    for p = 1:epoch
        % Loop over each core in x_qtt
        for i = 1:d1
            %%% Reshape F_subsampled into quantized format
            F_col_quantized = reshape(F_subsampled, [N, 2 * ones(1, d1)]);
            
            %%% contract the matrix tensor with x_qtt except for the ith core
            if i == 1
                x_qtt_after = tt_contraction(x_qtt, 2, d1, true, false); % Contract the cores after the ith core
        
                % contract matrix tensor with the x_qtt_after
                ind_mat_after = 3:ndims(F_col_quantized); % contract from n(2) to n(d1)
                ind_vec_after = 2:ndims(x_qtt_after);   % Indices for x_qtt_after
                A = tns_mult(F_col_quantized, ind_mat_after, x_qtt_after, ind_vec_after);
                % A now has index order: (N, n(1), r(2))
            elseif i == d1
                x_qtt_before = tt_contraction(x_qtt, 1, d1-1, false, true); % Contract the cores before the ith core
                
                % Contract the matrix tensor with the cores before the ith core
                ind_mat_before = 2:d1;        % Contract from n(1) to n(i-1), shifted by +1 for row index N
                ind_vec_before = 1:(d1-1);    % Indices for x_qtt_before
                A = tns_mult(F_col_quantized, ind_mat_before, x_qtt_before, ind_vec_before); 
                % A now has index order: (N, n(d1), r(d1))
                
                % permute the tensor to get index order (N, r(d1), n(d1))
                A = permute(A, [1, 3, 2, 4]);
            else
                x_qtt_before = tt_contraction(x_qtt, 1, i-1, false, true); % Contract the cores before the ith core
                x_qtt_after = tt_contraction(x_qtt, i+1, d1, true, false); % Contract the cores after the ith core
                
                % Contract the matrix tensor with the cores before the ith core
                ind_mat_before = 2:i;  % % Contract from n(1) to n(i-1), shifted by +1 for row index N
                ind_vec_before = 1:(i-1);      % Indices for x_qtt_before
                A = tns_mult(F_col_quantized, ind_mat_before, x_qtt_before, ind_vec_before); 
                % A now has index order: (N, n(i), n(i+1), ..., n(d1), r(i))
        
                % contract matrix tensor with the x_qtt_after
                ind_mat_after = 3:d1-i+2; % contract from n(i+1) to n(d1)
                ind_vec_after = 2:ndims(x_qtt_after);   % Indices for x_qtt_after
                A = tns_mult(A, ind_mat_after, x_qtt_after, ind_vec_after);
                % A now has index order: (N, n(i), r(i), r(i+1))
                
                % permute the tensor to get index order (N, r(i), n(i), r(i+1))
                A = permute(A, [1, 3, 2, 4]);
            end
        
            %%% update the ith core
            A = reshape(A, [N, numel(A)/N]);
            x = solve_normal_equation(A, y);
        
            ps = x_qtt.ps;  
            x_qtt.core(ps(i):ps(i+1)-1) = x;
        
        end
    
        % Compute and store the norm difference
        error_list(p) = norm(full(x_qtt) - x_benchmark) / norm(x_benchmark);
        % fprintf('Epoch %d: Error = %.5f\n', p, error_list(p));
    end

    disp('After optimization:');
    disp(error_list(end));
end

function x = solve_normal_equation(A, b)
% SOLVE_NORMAL_EQUATION Solves a least-squares problem using the normal equation.
%   Method:
%       The least-squares minimization is equivalent to solving the normal equation:
%           A^T A x = A^T b
    % Input validation
    if size(A, 1) ~= size(b, 1)
        error('The number of rows in A must match the length of b.');
    end

    % Solve the normal equation
    % % warning on
    % x = (A' * A) \ (A' * b);
    % warning off
    warning('off', 'MATLAB:nearlySingularMatrix'); % Turn off specific warning
    x = (A' * A) \ (A' * b);
    warning('on', 'MATLAB:nearlySingularMatrix'); % Turn the warning back on
end