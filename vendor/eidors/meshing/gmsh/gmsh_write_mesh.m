function gmsh_write_mesh(mdl, data, outfile)
%gmsh_read_mesh(mdl, [data,] outfile)
% Write a GMSH .msh file (v4.1 format)
%
% mdl      An EIDORS model (fwd_model or rec_model) or an image (img)
% data     Per element data (optional)
% outfile  Output filename, should end in '.msh'
%
% If 'mdl' is an EIDORS image, then img.elem_data will be used if 'data' is not
% provided.

% (C) 2021 Alistair Boyle. Licensed under GPL v2 or v3
% Modified by Bartek Grychtol
% $Id: gmsh_write_mesh.m 7102 2024-12-26 12:12:18Z aadler $

if ischar(mdl) && strcmp(mdl,'UNIT_TEST'); do_unit_test; return; end

is_img = (isfield(mdl, 'type') && strcmp(mdl.type, 'image')) || isfield(mdl, 'elem_data');
is_imdl = (isfield(mdl, 'type') && strcmp(mdl.type, 'inv_model')) || isfield(mdl, 'fwd_model');

if nargin < 3
    outfile = data;
    data = [];
end

if is_img && isempty(data)
    data = mdl.elem_data(:,1); % TODO only handles first frame of data at the moment
end
assert(isempty(data) || size(data,2) == 1, 'error: expect a single frame of data in gmsh_write_mesh');

if is_imdl || is_img
    if isfield(mdl, 'rec_model')
        mdl = mdl.rec_model;
    else
        mdl = mdl.fwd_model;
    end
end
assert(isempty(data) || size(data,1) == size(mdl.elems,1), 'error: expect a data to match number of elements in gmsh_write_mesh');

nn = size(mdl.nodes,1); % number of nodes
ne = size(mdl.elems,1); % number of elements
ndim = size(mdl.nodes,2); % number of dimensions: 2=2D or 3=3D
assert((ndim >= 2) && (ndim <= 3), 'not 2D or 3D?!');

fid = fopen(outfile, 'w');
fprintf(fid, '$MeshFormat\n');
fprintf(fid, '4.1 0 8\n'); % version file-type(0 for ASCII, 1 for binary) data-size=sizeof(size_t)
fprintf(fid, '$EndMeshFormat\n');

nodes = [ mdl.nodes zeros(nn, 3 - ndim) ]; % always nn x 3, with z=0.0 if 2D


[mdl, mat_vec] = prepare_materials(mdl, ne);
numEntityBlocks = 1;
try numEntityBlocks = numel(mdl.mat_idx); end
numPhysicalNames = 0;
try numPhysicalNames = numel(mdl.mat_idx); end

if numPhysicalNames
   fprintf(fid, '$PhysicalNames\n');
   fprintf(fid, '%d\n',numPhysicalNames);
   for i = 1:numPhysicalNames
      fprintf(fid, '%d %d "%s"\n', ndim, i, mdl.mat_names{i}); % dimension physicalTag "name"
   end
   fprintf(fid, '$EndPhysicalNames\n');
end

fprintf(fid, '$Entities\n');
if ndim == 2
   fprintf(fid, '%d %d %d %d\n', 0, 0, numEntityBlocks, 0); % numPoints numCurves numSurfaces numVolumes
else
   fprintf(fid, '%d %d %d %d\n', 0, 0, 0, numEntityBlocks); % numPoints numCurves numSurfaces numVolumes
end
for i = 1:numEntityBlocks
   physicalTag = 0; 
   if numPhysicalNames, physicalTag = i; end
   if numEntityBlocks == 1
      idx = (1:ne)';
   else 
      idx = reshape(mdl.mat_idx{i},[],1);
   end
   minlim = min(nodes(unique(mdl.elems(idx,:)), :));
   maxlim = max(nodes(unique(mdl.elems(idx,:)), :));
   fprintf(fid, '%d %f %f %f %f %f %f %d %d %d\n', ...  
            i, ... % volumeTag
            minlim, ... % minX minY minZ 
            maxlim, ... % maxX maxY maxZ
            1, physicalTag, 0); % numPhysicalTags physicalTag numBoundingSurfaces
end
fprintf(fid, '$EndEntities\n');

fprintf(fid, '$Nodes\n');
fprintf(fid, '%d %d %d %d\n', 1, nn, 1, nn); % numEntityBlocks numNodes minNodeTag maxNodeTag
fprintf(fid, '%d %d %d %d\n', ndim, 0, 0, nn); % entityDim entityTag parametric(0 or 1) numNodesInBlock
fprintf(fid, '%d\n', 1:nn); % nodeTag
fprintf(fid, '%f %f %f\n', nodes'); % x y z
fprintf(fid, '$EndNodes\n');

fprintf(fid, '$Elements\n');
if ndim == 2
    type = 2; % triangle
else
    type = 4; % tetrahedra
end
fprintf(fid, '%d %d %d %d\n', numEntityBlocks, ne, 1, ne); % numEntityBlocks numElements minElementTag maxElementTag

for i = 1:numEntityBlocks
   if isempty(mat_vec)
      neb = ne;
      idx = (1:ne)';
   else
      neb = numel(mdl.mat_idx{i});
      idx = mdl.mat_idx{i};
   end
   fprintf(fid, '%d %d %d %d\n', ndim, i, type, neb); % entityDim entityTag elementType(4=tet) numElementsInBlock
   elems = [idx mdl.elems(idx,:)];
   if ndim == 2
      fprintf(fid, '%d %d %d %d\n', elems'); % elementTag nodeTag ...
   else
      fprintf(fid, '%d %d %d %d %d\n', elems'); % elementTag nodeTag ...
   end
end
fprintf(fid, '$EndElements\n');

if ~isempty(data)
    fprintf(fid, '$ElementData\n');
    fprintf(fid, '%d\n', 1); % numStringTags
    name = 'gmsh_write_mesh output';
    if isfield(mdl, 'name')
        name = mdl.name;
    end
    fprintf(fid, '%s\n', name); % stringTag
    fprintf(fid, '%d\n', 1); % numRealTags
    fprintf(fid, '%f\n', 0); % realTag(time=0)
    fprintf(fid, '%d\n', 3); % numIntegerTags
    fprintf(fid, '%d\n', 0); % integerTag(time-step=0)
    fprintf(fid, '%d\n', 1); % integerTag(1-component/scalar field)
    fprintf(fid, '%d\n', ne); % integerTag(associated element values)
    data = [[1:ne]' data];
    fprintf(fid, '%d %f\n', data'); % integerTag(associated element values)
    fprintf(fid, '$EndElementData\n');
end
fclose(fid);

function [mdl, mat_vec] = prepare_materials(mdl, ne)
mat_vec = [];
if isfield(mdl, 'mat_idx') && ~isempty(mdl.mat_idx)
   mat_vec = zeros(ne,1);
   for i = 1:numel(mdl.mat_idx)
      idx = mdl.mat_idx{i};
      if any(mat_vec(idx))
         error('Materials are not unique');
      end
      mat_vec(idx) = i;
   end
   if isfield(mdl,'mat_names') && ~isempty(mdl.mat_names)
      assert(numel(mdl.mat_names) == numel(mdl.mat_idx),'Every mat_idx must have a name!')
   end
   if any(mat_vec==0)
      mdl.mat_idx(end+1) = {find(mat_vec==1)};
      if isfield(mdl,'mat_names')
         mdl.mat_names(end+1) = {'Background'};
      end
   end
end


function do_unit_test()
% TODO cannot check 'data' because gmsh_read_mesh and gmsh_mk_fwd_model
%      don't know about ElementData regions in the .msh file
outfile = tempname();
selfdir = fileparts(which('gmsh_read_mesh'));
mdl1 = gmsh_mk_fwd_model(fullfile(selfdir, 'tests/cube-4.1.msh'));
mdl2 = gmsh_mk_fwd_model(fullfile(selfdir, 'tests/box-4.1.msh'));
data1 = rand(size(mdl1.elems,1), 1);
data2 = rand(size(mdl1.elems,1), 1);
data3 = rand(size(mdl2.elems,1), 1);

gmsh_write_mesh(mdl1, data1, outfile);
mdl = gmsh_mk_fwd_model(outfile);
delete(outfile);
unit_test_cmp('fmdl+data (nn)', mdl1.nodes, mdl.nodes);
unit_test_cmp('fmdl+data (el)', mdl1.elems, mdl.elems);
% TODO check 'data' matches data1

gmsh_write_mesh(mdl1, outfile);
mdl = gmsh_mk_fwd_model(outfile);
delete(outfile);
unit_test_cmp('fmdl (nn)', mdl1.nodes, mdl.nodes);
unit_test_cmp('fmdl (ee)', mdl1.elems, mdl.elems);
% TODO check 'data' does not exist

img = mk_image(mdl1, data2);
gmsh_write_mesh(img, data1, outfile);
mdl = gmsh_mk_fwd_model(outfile);
delete(outfile);
unit_test_cmp('img+data (nn)', mdl1.nodes, mdl.nodes);
unit_test_cmp('img+data (ee)', mdl1.elems, mdl.elems);
% TODO check 'data' matches data1

img = mk_image(mdl1, data2);
gmsh_write_mesh(img, outfile);
mdl = gmsh_mk_fwd_model(outfile);
delete(outfile);
unit_test_cmp('img (nn)', mdl1.nodes, mdl.nodes);
unit_test_cmp('img (ee)', mdl1.elems, mdl.elems);
% TODO check 'data' matches data2

imdl = select_imdl(mdl1, {'Basic GN dif'});

gmsh_write_mesh(imdl, outfile);
mdl = gmsh_mk_fwd_model(outfile);
delete(outfile);
unit_test_cmp('imdl+fmdl (nn)', mdl1.nodes, mdl.nodes);
unit_test_cmp('imdl+fmdl (ee)', mdl1.elems, mdl.elems);
% TODO check 'data' does not exist

gmsh_write_mesh(imdl, data1, outfile);
mdl = gmsh_mk_fwd_model(outfile);
delete(outfile);
unit_test_cmp('imdl+fmdl+data (nn)', mdl1.nodes, mdl.nodes);
unit_test_cmp('imdl+fmdl+data (ee)', mdl1.elems, mdl.elems);
% TODO check 'data' matches data1

imdl.rec_model = mdl2;

gmsh_write_mesh(imdl, outfile);
mdl = gmsh_mk_fwd_model(outfile);
delete(outfile);
unit_test_cmp('imdl+rmdl (nn)', mdl2.nodes, mdl.nodes);
unit_test_cmp('imdl+rmdl (ee)', mdl2.elems, mdl.elems);
% TODO check 'data' does not exist

gmsh_write_mesh(imdl, data3, outfile);
mdl = gmsh_mk_fwd_model(outfile);
delete(outfile);
unit_test_cmp('imdl+rmdl+data (nn)', mdl2.nodes, mdl.nodes);
unit_test_cmp('imdl+rmdl+data (ee)', mdl2.elems, mdl.elems);
% TODO check 'data' matches data3
