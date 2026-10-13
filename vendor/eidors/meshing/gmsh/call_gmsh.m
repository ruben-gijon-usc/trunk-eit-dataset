function status= call_gmsh(geo_file, extra)
% call_gmsh: call Gmsh to create a vol_file from a geo_file
% status= call_gmsh(geo_file, dim, extra)
%  staus = 0 -> success , negative -> failure
%
% geo_file = geometry file (input)
% extra    = additional string options for gmsh (see gmsh manual at
%            http://geuz.org/gmsh/doc/texinfo/gmsh.html#Command_002dline-options
%
% NOTE: Make sure the geo file contains the Mesh command!
%
% To debug the shapes and view the gmsh interface, use
%   eidors_debug on call_gmsh


% (C) 2009-2024 Bartosz Sawicki and Bartlomiej Grychtol
% License: GPL  version 2
% $Id: call_gmsh.m 6837 2024-05-02 08:29:02Z bgrychtol $

% default to 2-D model
if nargin<2
    dim = 2;
end
if nargin<3
   extra = [];
end

isoctave = getfield(eidors_obj('interpreter_version'),'isoctave');

cache_path = eidors_cache('cache_path');

% Gmsh executable filename
if  ~ispc
   % Matlab plays with libraries
   gmsh_name = 'LD_LIBRARY_PATH="" gmsh';
else
   gmsh_name = [cache_path,'/gmsh'];
end

% Matlab reports physical cores, Octave reports logical ones
num_threads = max(maxNumCompThreads -1 -isoctave, 2);
cmd = sprintf( '%s "%s" -nt %d %s',  gmsh_name, geo_file, ...
               num_threads, extra);
if ~eidors_debug('query','call_gmsh')
   cmd = [cmd ' -term -v 2 -save'];
end
while( 1 )

    status= system_cmd(cmd);

    if status==0; break; end

    if ~ispc
        disp(['It seems you are running Linux or MacOS and Gmsh has not worked. ' ...
            'Check that it is installed and on the path. \n' ...
            'Perhaps LD_LIBRARY_PATH needs to be set?' ]);
        break;
    else
        fprintf([ ...
            'Gmsh call failed. Is Gmsh installed and on the search path?\n' ...
            'You are running under windows, I can attempt to create\n' ...
            'a batch file to access gmsh.\n' ...
            'Please enter the directory in which to find gmsh.\n' ...
            'If you dont have a copy, download it from ' ...
            'http://www.geuz.org/gmsh/\n\n' ]);

        gmsh_path = input('gmsh_path? [or i=ignore, e=error] ','s');
        if strcmp(gmsh_path,'i'); break;end
        if strcmp(gmsh_path,'e'); error('user requested'),end;
        if exist( sprintf('%s/gmsh.exe',gmsh_path) , 'file' ) || ...
                exist( sprintf('%s/bin/gmsh.exe',gmsh_path) , 'file' )
            disp('Found gmsh');
            gmsh_exe = gmsh_path;
            if exist( sprintf('%s/bin/gmsh.exe',gmsh_path) , 'file' )
                gmsh_exe = [gmsh_path '/bin'];
            end


            fid= fopen([cache_path, '/gmsh.bat'],'w');
            fprintf(fid,'"%s/gmsh.exe" %%*\n', gmsh_exe);
            fclose(fid);
        end
    end
end
