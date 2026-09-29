function setup_qtt_paths(tt_toolbox_dir)
%SETUP_QTT_PATHS Add this repository and TT-Toolbox to the MATLAB path.
%
%   SETUP_QTT_PATHS verifies an existing TT-Toolbox installation.
%   SETUP_QTT_PATHS(TT_TOOLBOX_DIR) first adds the supplied installation
%   directory and its subdirectories to the MATLAB path.

code_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(code_dir, 'src', 'tensor'));
addpath(fullfile(code_dir, 'src', 'recovery'));

if nargin >= 1 && ~isempty(tt_toolbox_dir)
    if ~isfolder(tt_toolbox_dir)
        error('setup_qtt_paths:MissingDirectory', ...
            'TT-Toolbox directory not found: %s', tt_toolbox_dir);
    end
    addpath(genpath(tt_toolbox_dir));
end

required_names = {'tt_tensor', 'tt_random', 'my_chop2', ...
                  'fwht', 'ifwht'};
missing_names = required_names(cellfun(@(name) isempty(which(name)), ...
                                       required_names));

if ~isempty(missing_names)
    error('setup_qtt_paths:MissingDependencies', ...
        'Missing required MATLAB functions: %s', ...
        strjoin(missing_names, ', '));
end

fprintf('QTT compressed-sensing paths are ready.\n');
end
