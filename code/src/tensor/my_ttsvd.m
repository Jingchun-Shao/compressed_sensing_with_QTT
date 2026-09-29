function t = my_ttsvd(varargin)
%MY_TTSVD Constructor for TT-tensor.
%
%   T = TT_TENSOR(X, n, tt_rank) converts the full tensor X into a TT-tensor
%   with mode sizes specified by n and with TT-ranks truncated in each SVD step
%   so that they do not exceed tt_rank. The conversion accuracy is fixed at 1e-14.
%
%   Optionally, you can also specify the tailing ranks:
%       T = TT_TENSOR(X, n, tt_rank, r0, r_end)
%   where r0 and r_end are the first and last TT-ranks, respectively.
%
%   Inputs:
%       X       - Full format tensor.
%       n       - Size vector (mode sizes) for the tensor.
%       tt_rank - Maximum TT-rank allowed at each SVD truncation, a vector
%                 of length d+1 (where d = numel(n)).
%       r0      - (Optional) First TT-rank.
%       r_end   - (Optional) Last TT-rank.
%
%   Output:
%       t       - TT-tensor structure with fields: d, n, r, ps, core, over.

if is_array(varargin{1})
    t = tt_tensor;  % Create an empty tt_tensor structure.
    b = varargin{1};
    
    % Mode sizes n from the second argument (default to size(b) if missing)
    if (nargin >= 2 && is_array(varargin{2}) && ~isempty(varargin{2}))
        n = varargin{2};
    else
        n = size(b);
    end
    n = n(:);
    d = numel(n);
    
    % TT-rank threshold for each SVD step from the third argument.
    if (nargin >= 3 && ~isempty(varargin{3}))
        tt_rank = varargin{3};
    else
        tt_rank = Inf; % No additional truncation if not provided.
    end

    % Initialize TT-ranks as ones.
    r = ones(d+1, 1);
    % Optionally set the first and last ranks if provided.
    if (nargin >= 4 && is_array(varargin{4}) && numel(varargin{4}) == 1)
        r(1) = varargin{4};
    end
    if (nargin >= 5 && is_array(varargin{5}) && numel(varargin{5}) == 1)
        r(d+1) = varargin{5};
    end
    
    % Handle singleton or sparse tensor case.
    if ((numel(n)==2 && min(n)==1) || (numel(n)==1) || issparse(b))
        r = [r(1); r(d+1)];
        d = 1; n = prod(n);
        core = b(:);
        ps = cumsum([1; n .* r(1:d) .* r(2:d+1)]);
        t.d = d;
        t.n = n;
        t.r = r;
        t.ps = ps;
        t.core = core;
        t.over = 0;
        return;
    end
    
    c = b;
    if isa(b, 'gpuArray')
        core = gpuArray([]);
    else
        core = [];
    end
    pos = 1;
    % Fixed accuracy for conversion.
    eps_val = 1e-14;
    ep = eps_val / sqrt(d-1);
    
    % Main loop: perform SVD-based TT decomposition.
    for i = 1:d-1
        m = n(i) * r(i);
        c = reshape(c, [m, numel(c)/m]);
        [u, s, v] = svd(c, 'econ');
        s = diag(s);
        % Determine the numerical rank based on the tolerance.
        r1 = my_chop2(s, ep * norm(s));
        % Further threshold the rank using the provided tt_rank vector.
        r1 = min(r1, tt_rank(i+1));
        u = u(:, 1:r1); 
        s = s(1:r1);
        r(i+1) = r1;
        core(pos:pos + r(i)*n(i)*r(i+1) - 1) = u(:);
        v = v(:, 1:r1);
        v = v * diag(s);
        c = v';
        pos = pos + r(i)*n(i)*r(i+1);
    end
    core(pos:pos + r(d)*n(d)*r(d+1) - 1) = c(:);
    core = core(:);
    ps = cumsum([1; n .* r(1:d) .* r(2:d+1)]);
    
    % Populate the TT-tensor structure.
    t.d = d;
    t.n = n;
    t.r = r;
    t.ps = ps;
    t.core = core;
    return;
end
