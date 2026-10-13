function [srf,vtx,fc,bc,simp,edg,mat_ind,phys_names,entities] = gmsh_read_mesh(filename)
%[srf,vtx,fc,bc,simp,edg,mat_ind,phys_names,entities] = gmsh_read_mesh(filename)
% Function to read in a mesh model from Gmsh and saves it in
% five arrays; surface (srf), veritices (vtx), face no. (fc)
% volume (simp) and edges (edg)
%
% srf        = The surfaces indices into vtx
% simp       = The volume indices into vtx
% vtx        = The vertices matrix
% fc         = A one column matrix containing the face numbers
% edg        = Edge segment information
% filename   = Name of file containing NetGen .vol information
% mat_ind    = Material index
% phys_names = Structure of "Physical" entities in the mesh
%              .dim   = dimension
%              .name  = name (string)
%              .tag   = physical tag
%              .nodes = N-x-dim array of indices into vtx
%
% This mostly works on GMSH v2. A very basic GMSH v4 reader is now
% included.

% $Id: gmsh_read_mesh.m 6848 2024-05-04 22:20:11Z bgrychtol $
% (C) 2009 Bartosz Sawicki. Licensed under GPL V2
% Modified by James Snyder, Mark Campbell, Symon Stowe, Alistair Boyle, 
% Bartek Grychtol

if ischar(filename) && strcmp(filename,'UNIT_TEST'); do_unit_test; return; end

fid = fopen(filename,'r');
content = textscan(fid,'%s','Delimiter','\n');
content = content{:};
fclose(fid);
phys_names = [];
entities = [];
idx = find(startsWith(content,'$'));
for i = 1:2:numel(idx)
   block = content( (idx(i)+1) : idx(i+1)-1 );
   switch content{idx(i)}
      case '$MeshFormat'
         gmshformat = parse_format(block);
      case '$Entities'
         entities= parse_entities( block, gmshformat );
      case '$Elements'
         elements= parse_elements( block, gmshformat );
      case '$Nodes'
         nodes= get_lines_with_nodes( block, gmshformat );
      case '$PhysicalNames'
         phys_names= parse_names( block, gmshformat );
   end
end

% look up nodes for each of phys_names
if ~isempty(phys_names)
    if (gmshformat >= 4.0)
        assert(~isempty(entities), 'expected $Entities section (GMSH format 4)');
        for i = 1:length(phys_names)
            phys_tag = phys_names(i).tag;
            dim = phys_names(i).dim;
            tmpentities = find(arrayfun(@(x) any(x.phys_tag == phys_tag), entities));
            tags = cat(1,entities(tmpentities).entity_tag);
            idx = ismember(elements.entity_tag, tags) & elements.dim == dim;
            phys_names(i).nodes = elements.simp(idx,1:dim+1);
            elements.phys_tag(idx) = phys_tag;
        end
    else % GMSH 2.0 doesn't have Entities
        assert(isempty(entities), 'unexpected $Entities section (GMSH format 2)');
        for i = 1:length(phys_names)
            dim = phys_names(i).dim;
            tag = phys_names(i).tag;
            idx = elements.phys_tag == tag & elements.dim == dim;
            phys_names(i).nodes = elements.simp(idx,1:dim+1);
        end
    end
end

edg = [];
bc = [];

% Type 2: 3-node triangle
tri = elements.type==2;
% Type 4: 4-node tetrahedron
tet = elements.type==4;

% Select 2d vs 3d model by checking if Z is all the same
if any(nodes(:,4) ~= nodes(1,4))
    vtx = nodes(:,2:4);
    simp = elements.simp(tet,:);
    srf = elements.simp(tri,1:3);
    mat_ind = elements.phys_tag(tet);
else
    vtx = nodes(:,2:3);
    simp = elements.simp(tri,1:3);
    srf = [];
    mat_ind = elements.phys_tag(tri);
end

fc = elements.phys_tag(tri); % for 4.1, these are all zero, why?
end

function mat = get_lines_with_nodes( lines, gmshformat )
	
	
    switch floor(gmshformat)
    % Version 2 Line Format:
    % node-number x-coord y-coord z-coord
    % Version 4 Line Format: (not always like this)
    % node-number
    % x-coord y-coord z-coord
	case 2 
      n_rows = parse_rows(lines{1},gmshformat);
      %mat= sscanf(sprintf('%s ',lines{2:end}),'%f',[4,n_rows])';
      mat= textscan(sprintf('%s ',lines{2:end}),'%f');
      mat = reshape(mat{1},[4,n_rows])';
	case 4;
        n = sscanf(lines{1}, '%d')';
        n_block = n(1);
        n_nodes = n(2);
        mat = zeros(n_nodes,4);
        count = 2;
        while n_block > 0
            n_block = n_block - 1;
            tline = lines{count}; count = count + 1; % fgetl(fid);
            blk = sscanf(tline, '%d')';
            blk_nodes = blk(end); % n(2) for v4.0, n(4) for v4.1
            if blk_nodes == 0,  continue, end
            if gmshformat == 4.1 % v4.1: node tags first as a block, then node coordinates in a block
               node_tag =  textscan(sprintf('%s ',lines{count:count+blk_nodes-1}),'%f');
               node_tag = node_tag{1};
               count = count + blk_nodes;
%                el = sscanf(sprintf('%s ',lines{count:count+blk_nodes-1}),'%f',[3 blk_nodes])';
               el = textscan(sprintf('%s ',lines{count:count+blk_nodes-1}),'%f');
               el = reshape(el{1},[3 blk_nodes])';
               count = count + blk_nodes;
               mat(node_tag,:) = [node_tag(:), el(:,1:end)];
            else % v4.0: node tag and coordinates on the same line
%                data = sscanf(sprintf('%s ',lines{count:count+blk_nodes-1}),'%f',[4 blk_nodes])';
               data = textscan(sprintf('%s ',lines{count:count+blk_nodes-1}),'%f');
               data = reshape(data{1},[4 blk_nodes])';
               mat(data(:,1),:) = data;
               count = count + blk_nodes;               
            end
            
        end
        assert(n_block == 0, 'failed to consume all $Nodes blocks');
    otherwise; error('cant parse gmsh file of this format');
	end
end

function gmshformat = parse_format(lines)
   rawformat = sscanf(lines{1},'%f');
   gmshformat = rawformat(1);
end

function n_rows = parse_rows(tline, gmshformat)
   n_rows = sscanf(tline,'%d');
   switch floor(gmshformat)
     case 2; n_rows = n_rows(1);
     case 4; n_rows = n_rows(2);
     otherwise; error('cant parse gmsh file of this format');
   end
end

function names = parse_names( lines, version )
    % Line Format:
    % physical-dimension physical-number "physical-name"
    n_rows = sscanf(lines{1},'%d');
    names = struct('tag',{},'dim',{},'name',{});
    for i = 1:n_rows
        tline = lines{1+i};
        parts = regexp(tline,' ','split');
        nsz = size(names,2)+1;
        names(nsz).dim = str2double( parts(1) );
        names(nsz).tag = str2double( parts(2) );
        tname = parts(3);
        names(nsz).name = strrep(tname{1},'"','');
    end
end % end function

function elements = parse_entities( lines, gmshformat )
    elements = [];
    switch floor(gmshformat)
      case 2; warning('ignoring $Entities$ for GMSH v2 file');
      case 4;
          elements = parse_v4_entities(lines, gmshformat);
      otherwise error('cant parse this file type');
    end
 end

 function entities = parse_v4_entities(lines, gmshformat)
% http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
% $Entities
%  numPoints(size_t) numCurves(size_t)
%    numSurfaces(size_t) numVolumes(size_t)
%  pointTag(int) X(double) Y(double) Z(double)
%    numPhysicalTags(size_t) physicalTag(int) ...
%  ...
%  curveTag(int) minX(double) minY(double) minZ(double)
%    maxX(double) maxY(double) maxZ(double)
%    numPhysicalTags(size_t) physicalTag(int) ...
%    numBoundingPoints(size_t) pointTag(int) ...
%  ...
%  surfaceTag(int) minX(double) minY(double) minZ(double)
%    maxX(double) maxY(double) maxZ(double)
%    numPhysicalTags(size_t) physicalTag(int) ...
%    numBoundingCurves(size_t) curveTag(int) ...
%  ...
%  volumeTag(int) minX(double) minY(double) minZ(double)
%    maxX(double) maxY(double) maxZ(double)
%    numPhysicalTags(size_t) physicalTag(int) ...
%    numBoundngSurfaces(size_t) surfaceTag(int) ...
%  ...
% $EndEntities
% Entities are necessary to map '$PhysicalNames' to 'entityDim' & 'entityTag'
% in $Elements and $Nodes.
% Note that the entityTag can repeat for each type of entityDim, so when we
% look up the elements associated with a PhysicalName we need to then use the
% entityDim AND entityTag to find matching Nodes/Elements.
    entities = struct('phys_tag',{},'dim',{},'entity_tag',{});

    bl = sscanf(lines{1}, '%d')'; % Get the line info
    n_points = bl(1);
    n_curves = bl(2);
    n_surf = bl(3);
    n_vol = bl(4);
    %fprintf('entities: %d pts, %d curves, %d surf, %d vol\n', n_points, n_curves, n_surf, n_vol);
    for i = 2:numel(lines)
        tline = lines{i};
        bl = sscanf(tline, '%f')'; % Get the line info
        off = 8; % offset for 'numPhysicalTags'
        if n_points > 0
            n_points = n_points - 1;
            dim = 0; % point
            off = 5; % offset for 'numPhysicalTags'
        elseif n_curves > 0
            n_curves = n_curves - 1;
            dim = 1; % line/curve
        elseif n_surf > 0
            n_surf = n_surf - 1;
            dim = 2; % surface
        elseif n_vol > 0
            n_vol = n_vol - 1;
            dim = 3; % volume
        else
            error('extra $Entities found in v4.1 GMSH file');
        end
        if bl(off) > 0
            entities(end+1).phys_tag = bl((off+1):uint32(off+bl(off)));
            entities(end).entity_tag = uint32(bl(1));
            entities(end).dim = dim;
        end
    end
    assert(n_points == 0, 'missing Entity/points from GMSH 4.1 file');
    assert(n_curves == 0, 'missing Entity/curves from GMSH 4.1 file');
    assert(n_surf == 0, 'missing Entity/surfaces from GMSH 4.1 file');
    assert(n_vol == 0, 'missing Entity/volumes from GMSH 4.1 file');
end


function elements = parse_elements( lines, gmshformat )
   n_rows = parse_rows(lines{1},gmshformat);
   switch floor(gmshformat)
     case 2; elements = parse_v2_elements(lines(2:end),n_rows);
     case 4; elements = parse_v4_elements(lines,gmshformat);
     otherwise error('cant parse this file type');
   end
end

function elements = parse_v4_elements(lines,gmshformat)
% http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
% $Elements
% numEntityBlocks numElements minElementTag maxElementTag
% ...
% entityDim entityTag elementType numElementsInBlock
% elementTag nodeTag ... nodeTag
% ...
% $EndElements
% An entity is a node, curve, surface or volume
% Each entity has a list of contained elements and corresponding node tags
% 0D - 1 node tag (ignore these)
% 1D - 2 node tags
% 2D - 3 node tags
% 3D - 4 node tags
 
    bl = sscanf(lines{1}, '%d')'; % Get the line info
    n_blocks = bl(1);
    n_elems = bl(2);
    e_block = 0;
    simp = zeros([n_elems,4],'uint32');
    type = zeros([n_elems,1],'uint8');
    entity_tag = zeros([n_elems,1],'uint32');
    type = zeros([n_elems,1], 'uint8');
    dim = zeros([n_elems,1], 'uint8');
    phys_tag = zeros([n_elems,1], 'uint32'); 
    count = 2;
    for i = 1:n_blocks
        bl = sscanf(lines{count}, '%d')'; % Get the line info
        count = count + 1;
        if gmshformat >= 4.1
           e_dim = bl(1); % entityDim
           e_tag = bl(2); % entityTag
        else % 4.0
           e_tag = bl(1); % entityTag
           e_dim = bl(2); % entityDim
        end
        e_type = bl(3); % elementType
        e_block = bl(4); % numElementsInBlock: Size of entity block
%         data = sscanf(sprintf('%s ',lines{count:count+e_block-1}),'%d',[e_dim+2 e_block])';
        data = textscan(sprintf('%s ',lines{count:count+e_block-1}),'%d');
        data = reshape(data{1},[e_dim+2 e_block])';
        count = count + e_block;
        n_elems = n_elems - size(data,1);
        idx = data(:,1);
        simp(idx,1:e_dim+1) = data(:,2:end);
        type(idx) = e_type;
        entity_tag(idx) = e_tag;
        dim(idx) = e_dim;
    end
    assert(n_elems == 0, 'missing Elements from GMSH 4 file');
%     assert(n_blocks == 0, 'missing Element/blocks from GMSH 4 file');
    elements = struct('simp',simp,'type',type,'entity_tag',entity_tag,...
                      'dim',dim,'phys_tag',phys_tag);

% Copied from somewhere else in the code, to reintroduce once we deal
% with PhysTags again
%     assert(length(elements(j).phys_tag) > 0, ...
%        'GMSH format v4 mesh volume elements can only have one $PhysicalName when imported into EIDORS');
end

function elements = parse_v2_elements(lines,n_rows)
% Line Format:
% elm-number elm-type number-of-tags < tag > ... node-number-list
elements.simp = zeros([n_rows,4],'uint32');;
elements.phys_tag = zeros([n_rows,1], 'uint32');
elements.geom_tag = zeros([n_rows,1],'uint32');;
elements.type = zeros([n_rows,1],'uint8');
elements.dim = zeros([n_rows,1], 'uint8');

for i = 1:n_rows % can we get rid of this loop?
    tline = lines{i};
    n = sscanf(tline, '%d')';
    ndim = numel(n) - n(3) - 4;
    elements.simp(i,1:ndim+1) = n(n(3) + 4:end);
    elements.type(i) = n(2);
    elements.dim(i) = ndim;
    if n(3) > 0 % get tags if they exist
        % By default, first tag is number of parent physical entity
        % second is parent elementary geometrical entity
        % third is number of parent mesh partitions followed by
        % partition ids
        tags = n(4:3+n(3));
        if length(tags) >= 1
            elements.phys_tag(i) = tags(1);
            if length(tags) >= 2
                elements.geom_tag(i) = tags(2);
            end
        end
    end
end
end

function do_unit_test
   selfdir = fileparts(which('gmsh_read_mesh'));
   [srf,vtx,fc,bc,simp,edg,mat_ind,phys_names] = gmsh_read_mesh(fullfile(selfdir, 'tests/test-4.0.msh'));
    unit_test_cmp('v4 vtx ',vtx(2:3,:),[1,0;-1,0])
    unit_test_cmp('v4 simp',simp(2:3,:),[3,7,12; 12, 7,14]);

   [srf,vtx,fc,bc,simp,edg,mat_ind,phys_names] = gmsh_read_mesh(fullfile(selfdir, 'tests/test-2.2.msh'));
    unit_test_cmp('v2 vtx ',vtx(2:3,:),[1,0;-1,0])
    unit_test_cmp('v2 simp',simp(2:3,:),[2,4,15; 14,17,19]);

   vers = {'2.2', '4.0', '4.1'};
   for ver = vers(:)'
       ver = ver{1};
       [srf,vtx,fc,bc,simp,edg,mat_ind,phys_names] = ...
           gmsh_read_mesh( fullfile(selfdir, ['tests/box-' ver '.msh']) );
       unit_test_cmp(['2d v' ver ' vtx '],vtx,[0,0;1,0;1,1;0,1;0.5,0.5])
       unit_test_cmp(['2d v' ver ' simp'],simp,[2,5,1;1,5,4;3,5,2;,4,5,3]);
       unit_test_cmp(['2d v' ver ' phys'],{phys_names(:).name},{'elec#1','elec#2','main'});

       [srf,vtx,fc,bc,simp,edg,mat_ind,phys_names] = ...
           gmsh_read_mesh( fullfile(selfdir, ['tests/cube-' ver '.msh']) );
       unit_test_cmp(['3d v' ver ' vtx '],vtx([1,2,13,end],:),[0,0,1;0,0,0;,0.5,0.5,0;0.5,0.5,1])
       unit_test_cmp(['3d v' ver ' simp'],simp([1,2,23,end],:), ...
           [10,11,12,13;9,12,14,11;12,14,10,7;13,10,11,6]);
       unit_test_cmp(['3d v' ver ' phys'],{phys_names(:).name},{'elec#1','elec#2','main'});
   end

end
