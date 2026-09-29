function a = tt_contraction(tt, start_idx, end_idx, start_idx_keep, end_idx_keep)
% TT_CONTRACTION Contracts the cores of a TT from the start_idx core to the end_idx core.
% Optionally, you can choose whether to keep boundary dimensions of size 1 by setting
% start_idx_keep or end_idx_keep to true.
%
%   a = TT_CONTRACTION(tt, start_idx, end_idx) contracts the TT cores between the specified
%   start and end indices. By default, dimensions corresponding to TT ranks of 1 are removed.
%
%   a = TT_CONTRACTION(tt, start_idx, end_idx, start_idx_keep, end_idx_keep) allows you to specify
%   whether to keep the first and/or last boundary dimensions even if they are equal to 1.
%
% Input:
%   tt             - TT-structure with fields:
%                       .d    : Number of dimensions.
%                       .n    : Mode sizes for each core (vector).
%                       .ps   : Starting positions of each core in the flattened core array.
%                       .core : Flattened array of all TT cores.
%                       .r    : TT ranks (vector of length d+1).
%   start_idx      - Starting index of contraction (1-indexed).
%   end_idx        - Ending index of contraction (1-indexed).
%   start_idx_keep - (Optional) Boolean flag. If true, keep the first boundary dimension even if r(start_idx)==1.
%   end_idx_keep   - (Optional) Boolean flag. If true, keep the last boundary dimension even if r(end_idx+1)==1.
%
% Output:
%   a              - Contracted TT core, reshaped to a tensor with dimensions
%                    [r(start_idx), n(start_idx:end_idx), r(end_idx+1)] with optional removal
%                    of boundary dimensions when they equal 1.
%
% Example:
%   % Contract cores 2 through 4, removing the first boundary if rank==1, but keeping the last
%   a = tt_contraction(tt, 2, 4, false, true);

    % Set default values for optional flags if not provided
    if nargin < 4
        start_idx_keep = false;
    end
    if nargin < 5
        end_idx_keep = false;
    end

    % Extract TT parameters
    d    = tt.d;    % Number of dimensions
    n    = tt.n;    % Mode sizes for each core
    ps   = tt.ps;   % Starting positions of each core in the flattened core array
    core = tt.core; % Flattened cores
    r    = tt.r;    % TT ranks

    % Validate indices
    if start_idx < 1 || start_idx > d || end_idx < 1 || end_idx > d || start_idx > end_idx
        error('Invalid start_idx or end_idx. Ensure 1 <= start_idx <= end_idx <= d.');
    end

    %%% Special case: Contract a single core
    if end_idx == start_idx
        a = core(ps(start_idx):ps(start_idx+1)-1);

        % Reshape the result back to the desired tensor form
        result_dims = [r(start_idx), n(start_idx).', r(end_idx+1)];
        
        % Remove unnecessary dimensions if the corresponding keep flag is false
        if ~start_idx_keep && r(start_idx) == 1
            result_dims(1) = [];  % Remove the first dimension
        end
        if ~end_idx_keep && r(start_idx+1) == 1
            result_dims(end) = [];  % Remove the last dimension
        end

        a = reshape(a, result_dims);
        return;
    end

    %%% General case:
    % Initialize with the start core
    a = core(ps(start_idx):ps(start_idx+1)-1);

    % Iterate through the specified range to contract cores
    for i = (start_idx+1):end_idx
        cr = core(ps(i):ps(i+1)-1);
        cr = reshape(cr, [r(i), n(i)*r(i+1)]);
        a = reshape(a, [numel(a)/r(i), r(i)]);
        a = a * cr;
    end

    % Reshape the result back to the desired tensor form
    result_dims = [r(start_idx), n(start_idx:end_idx).', r(end_idx+1)];
    
    % Conditionally remove unnecessary boundary dimensions
    if ~start_idx_keep && r(start_idx) == 1
        result_dims(1) = [];  % Remove first dimension if it equals 1 and flag is false
    end
    if ~end_idx_keep && r(end_idx+1) == 1
        result_dims(end) = [];  % Remove last dimension if it equals 1 and flag is false
    end
    
    % Reshape the result
    a = reshape(a, result_dims);
end
