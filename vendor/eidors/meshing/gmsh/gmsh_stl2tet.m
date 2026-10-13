function mdl = gmsh_stl2tet(stlfile, mesh_size, extra, nopt)
%GMSH_STL2TET creates a tetrahedral mesh from an stl file
% mdl = gmsh_stl2tet(stlfile, maxh, extra, nopt) where:
%        mdl - EIDORS fwd_model struct
%    stlfile - either:
%                 - a path to an stl file
%               OR
%                 - an EIDORS fwd_model with .nodes and .boundary or .elems
%               OR
%                 - a struct with .vertices and .faces
%  mesh_size - controls mesh size. Either:
%                 - scalar: maximum edge length 
%                   (default: [] matches surface)
%               OR
%                 - [SizeMax, DistMax]: Linear grading from a fine surface,
%                   achieving SizeMax at DistMax distance from surface.
%      extra - extra command line options to gmsh (default: [])
%
%       nopt - number of extra optimization runs (default: 0)
%
% If stlfile is a struct, stl_write will be called first and an STL file
% written in a temporary location. 
%
% CASHING: Calls are cashed iff stlfile is a struct.
%
% NOTE: Only one surface per file is allowed.
%
% See also CALL_GMSH, STL_WRITE

% (C) Bartlomiej Grychtol, 2012-2021.
% $Id: gmsh_stl2tet.m 6983 2024-11-14 20:21:00Z bgrychtol $

if nargin < 4, nopt = 0; end
if nargin < 3, extra = []; end
if nargin < 2, mesh_size = []; end

if isstruct(stlfile)
    try
       name = stlfile.name;
    catch
       name = 'gmsh_stl2tet';
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
    opt.cache_obj{3} = mesh_size;
    opt.cache_obj{4} = extra;
    opt.cache_obj{5} = nopt;
    mdl = eidors_cache(@do_gmsh_stl2tet,{stlfile, mesh_size, extra, nopt}, opt);
else
    [~, name, ext] = fileparts(stlfile); name = [name ext];
    mdl = do_gmsh_stl2tet(stlfile, mesh_size, extra, nopt);
end

mdl = eidors_obj('fwd_model', mdl, 'name', name);


function mdl = do_gmsh_stl2tet(stlfile, mesh_size, extra, nopt)
if isstruct(stlfile)
    stem = tempname;
    stl_write(stlfile, [stem, '.stl'])
    stlname = [stem '.stl'];
else
    stem = strrep(stlfile,'.stl','');
    stlname = stlfile;
end
%TODO: Some of this could be exposed as options (Algorithm, Optimize, ...)
fid = fopen([stem '.geo'],'w');
fprintf(fid,'General.Terminal=1;\n');
fprintf(fid,'Merge "%s";\n',stlname);
fprintf(fid,'Surface Loop(1) = {1};\n');
fprintf(fid,'Volume(2) = {1};\n');
fprintf(fid,'Physical Volume(3) = {2};\n');

if numel(mesh_size) == 1
   fprintf(fid,'Mesh.MeshSizeMax = %f;\n', mesh_size);
elseif numel(mesh_size) == 2
   % Only works for HPX - experimental!
   fprintf(fid, 'Field[1] = Extend;\n');
   fprintf(fid, 'Field[1].SurfacesList = {1};\n');
   fprintf(fid, 'Field[1].SizeMax = %g;\n', mesh_size(1));
   fprintf(fid, 'Field[1].DistMax = %g;\n', mesh_size(2));
   fprintf(fid, 'Field[1].Power = 1;\n');
   fprintf(fid, 'Background Field = 1;\n');
   fprintf(fid, 'Mesh.MeshSizeExtendFromBoundary = 0;\n');
   fprintf(fid, 'Mesh.Algorithm3D=10;\n'); % HPX (multi-threaded delaunay)
end
if numel(mesh_size) < 2
   fprintf(fid,'Mesh.Algorithm3D=4;\n'); % 4=frontal (netgen)
end
fprintf(fid,'Mesh.OptimizeNetgen=1;\n');
fprintf(fid,'Mesh 3;\n');
% It seems that if Gmsh chooses to optimize, it does Gmsh, followed by
% Netgen (because OptimizeNetgen=1). This can be repeated.
for i = 1:nopt
   fprintf(fid, 'OptimizeMesh "Gmsh";\n');
   fprintf(fid, 'OptimizeMesh "Netgen";\n');
end
% Gmsh optimization is quick and experience suggests it's worth running
% after Netgen (again).
fprintf(fid, 'OptimizeMesh "Gmsh";\n');
fclose(fid);

call_gmsh([stem '.geo'], extra);

mdl = gmsh_mk_fwd_model([stem '.msh'],[],[],[]);

delete([stem '.geo']);
delete([stem '.msh']);
if isstruct(stlfile)
    delete(stlname);
end