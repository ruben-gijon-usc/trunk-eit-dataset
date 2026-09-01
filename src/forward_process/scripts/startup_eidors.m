% startup.m — Initialize EIDORS.
% Searches for EIDORS in $EIDORS_PATH env var, then a hardcoded fallback.

function startup_eidors()
    eidors_env = getenv('EIDORS_PATH');
    if ~isempty(eidors_env) && exist(fullfile(eidors_env, 'startup.m'), 'file')
        run(fullfile(eidors_env, 'startup.m'));
        return;
    end

    fallback = '/home/ruben/Documentos/EIDORS_implementation/eidors_lib/eidors-v3.12/eidors/startup.m';
    if exist(fallback, 'file')
        run(fallback);
        return;
    end

    error('EIDORS not found. Set the EIDORS_PATH environment variable.');
end
