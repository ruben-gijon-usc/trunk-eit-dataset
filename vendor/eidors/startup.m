% CUSTOM EIDORS STARTUP FILE
% this must be a script so that octave can source it

eidors_dir = fileparts(mfilename('fullpath'));
orig_dir = pwd();
try
    cd(eidors_dir);
    eidors_startup();
    cd(orig_dir);
catch
    cd(orig_dir);
    [message, id] = lasterr;
    if strcmp(id, 'MATLAB:infoXmlValidationError')
        % An old version of MATLAB that we don't know how to write info.xml for
    else
        error(message);
    end
end