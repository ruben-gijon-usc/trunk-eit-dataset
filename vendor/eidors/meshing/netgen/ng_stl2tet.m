function mdl = ng_stl2tet(stlfile, varargin)
%NG_STL2TET creates a tetrahedral mesh from an stl file
% Netgen treats STL files as approximate and re-meshes the boundary.
% The operation of Netgen is influenced by parameters specified in the
% ng.opt file (see NG_WRITE_OPT). The following parameters were found to be
% useful (Netgen's default values):
%
% opt.options.curvaturesafety = 2.0;
% opt.stloptions.yangle = 30;
% opt.stloptions.contyangle = 20;
% opt.stloptions.edgecornerangle = 60;
% opt.stloptions.chartangle = 15;
% opt.stloptions.outerchartangle = 70;
%
% USAGE:
% mdl = ng_stl2tet(stlfile, ...) where:
%        mdl - EIDORS fwd_model struct
%    stlfile - either:
%                 - a path to an stl file (!! must be ASCII !!)
%               OR
%                 - an EIDORS fwd_model with .nodes and .boundary or .elems
%               OR
%                 - a struct with .vertices and .faces
%        ... - parameters to ng_write_opt
%
% If stlfile is a struct, stl_write will be called first and an STL file
% written in a temporary location. 
%
% CASHING: Calls are cashed iff stlfile is a struct.
%
% NOTE: Only one surface per file is allowed.
%
% See also CALL_NETGEN, STL_WRITE, NG_WRITE_OPT, GMSH_STL2TET

% (C) Bartlomiej Grychtol, 2012-2021.
% $Id: ng_stl2tet.m 7124 2024-12-29 15:18:32Z aadler $

if isstruct(stlfile)
    try
       name = stlfile.name;
    catch
       name = 'ng_stl2tet';
    end
    try
       opt.cache_obj{1} = stlfile.nodes;
       if isfield(stlfile, 'boundary')
          opt.cache_obj{2} = stlfile.boundary;
       else 
          opt.cache_obj{2} = stlfile.elems;
       end
    catch
       opt.cache_obj = {stlfile.vertices, stlfile.faces};
    end
    if nargin > 1
       opt.cache_obj = horzcat(opt.cache_obj, varargin);
    end
    mdl = eidors_cache(@do_ng_stl2tet,[{stlfile}, varargin(:)'], opt);
else
    [~, name, ext] = fileparts(stlfile); name = [name ext];
    mdl = do_ng_stl2tet(stlfile, varargin{:});
end

mdl = eidors_obj('fwd_model', mdl, 'name', name);


function mdl = do_ng_stl2tet(stlfile, varargin)
if isstruct(stlfile)
    stem = tempname;
    stlname = [stem '.stl'];
    stl_write(stlfile, stlname, 'txt');
else
    stem = strrep(stlfile,'.stl','');
    stlname = stlfile;
end
volfile = [stem, '.vol'];
if nargin > 1
   ng_write_opt(varargin{:});
end
call_netgen(stlname,volfile);
if nargin > 1
   delete('ng.opt'); % clean up
end
mdl=ng_mk_fwd_model(volfile,[],[],[],[]);
mdl = fix_boundary(mdl);

