% startup.m — Initialize EIDORS.
% Searches for EIDORS in $EIDORS_PATH env var, then a hardcoded fallback.

function startup_eidors()
    % 1. Check EIDORS_PATH environment variable
    eidors_env = getenv('EIDORS_PATH');
    if ~isempty(eidors_env)
        startup_file = fullfile(eidors_env, 'startup.m');
        if ~exist(startup_file, 'file')
            startup_file = fullfile(eidors_env, 'eidors', 'startup.m');
        end
        if exist(startup_file, 'file')
            run(startup_file);
            return;
        end
    end

    % 2. Check vendor/eidors inside the repository
    scripts_dir = fileparts(mfilename('fullpath'));
    repo_root = fullfile(scripts_dir, '..', '..', '..', '..');
    vendor_startup = fullfile(repo_root, 'vendor', 'eidors', 'startup.m');
    if exist(vendor_startup, 'file')
        run(vendor_startup);
        return;
    end

    error('EIDORS not found. Please set the EIDORS_PATH environment variable (e.g., export EIDORS_PATH=/path/to/eidors).');
end
