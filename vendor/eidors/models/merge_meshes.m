function out = merge_meshes(M1, varargin)
%MERGE_MESHES - merges two meshes sharing only boundary nodes
% MERGE_MESHES(M1,M2,T) merges M2 in M1 using threshold T for 
%     detecting corresponding nodes. The meshes must not overlap.
% MERGE_MESHES(M1,M2,M3,..., T) merges M2, M3, ... into M1 (sequentially)
%
% Note that the boundaries of the separate meshes will only be
% concatenated, as this visualises nicely. To calculate the correct
% boundary use FIND_BOUNDARY.
%
% See also FIND_BOUNDARY

% (C) Bartlomiej Grychtol and Andy Adler, 2013-2024. Licence: GPL v2 or v3
% $Id: merge_meshes.m 6984 2024-11-14 20:24:35Z bgrychtol $

if ischar(M1) && strcmp(M1,'UNIT_TEST'), run_unit_test; return; end

if nargin < 3  || isstruct(varargin{end})
   th = mean(std(M1.nodes))/length(M1.nodes);
   shapes = varargin;
else
   th = varargin{end};
   shapes = varargin(1:end-1);
end

if ~isfield(M1, 'mat_idx') || isempty(M1.mat_idx)
   idx = 1:uint32(length(M1.elems));
   M1.mat_idx = {idx(:)};
end


out = M1;
out.elems = uint32(out.elems);
try out.boundary = uint32(out.boundary); end
if ~isfield(out, 'boundary') || isempty(out.boundary)
   out.boundary = find_boundary(out);
end
   

for i = 1:length(shapes)
    if length(shapes) > 1
       msg = sprintf('Merging mesh %d/%d ... ',i,length(shapes));
    else
       msg = 'Merging meshes ... ';
    end
    progress_msg(msg);
    
    M2 = shapes{i};
    M2.elems = uint32(M2.elems);
    try M2.boundary = uint32(M2.boundary); end

    if ~isfield(M2, 'boundary') || isempty(M2.boundary)
        M2.boundary = find_boundary(M2);
    end
    if ~isfield(M2, 'mat_idx') || isempty(M2.mat_idx)
        M2.mat_idx = {1:uint32(length(M2.elems))};
    end
    
    if size(out.elems,2) == 3 % surface mesh
        B1 = unique(out.elems);
    else
        B1 = unique(find_boundary(out));
    end
    
    if size(M2.elems,2) == 3 %surface mesh
        B2 = unique(M2.elems);
    else
        B2 = unique(find_boundary(M2));
    end
    
    M1_num_nodes = uint32(size(out.nodes,1));
    M2_num_nodes = uint32(size(M2.nodes,1));
    
    excl_B1 = ~nodes_in_bounding_box(M2.nodes(B2,:), out.nodes(B1,:), th);
    excl_B2 = ~nodes_in_bounding_box(out.nodes(B1,:), M2.nodes(B2,:), th);
    
    if nnz(excl_B1) == numel(B1) || nnz(excl_B2) == numel(B2)
        out.nodes = [out.nodes; M2.nodes];
        nodemap = M1_num_nodes + (1:M2_num_nodes);
    else
        B1(excl_B1) = [];
        B2(excl_B2) = [];
        
        bnd_nodes = false(size(M2.nodes,1),1);
        bnd_nodes(B2) = true;
        
        % add not-B2 to M1
        out.nodes = [out.nodes; M2.nodes(~bnd_nodes,:)];
        % build nodemap for B2
        nodemap = zeros(M2_num_nodes,1,'uint32');
        nodemap(~bnd_nodes) = M1_num_nodes + (1:uint32(M2_num_nodes - length(B2)));
        % find closest points in B1
        
        mem_use = 256 * 1024^2; %MiB
        if ispc || exist('OCTAVE_VERSION','builtin') &&  isunix
           meminfo = memory;
           mem_use = meminfo.MaxPossibleArrayBytes / 4;
        end
        % compute distance matrix in chunks to prevent exhausting memory
        chunk_sz = floor(mem_use / 8 / numel(B1)); % mem_use / 8 bytes(double)
        n_chunks = ceil(numel(B2)/chunk_sz);
        m = zeros(1,numel(B2));
        p = zeros(1,numel(B2));
        
        for c = 1:n_chunks
            progress_msg(c/n_chunks);
            idx = ((c-1)*chunk_sz + 1) : min(c*chunk_sz, numel(B2));
            D = (out.nodes(B1,1) - M2.nodes(B2(idx),1)').^2;
            for dim = 2:size(out.nodes,2)
                D = D + (out.nodes(B1,dim) - M2.nodes(B2(idx),dim)').^2;
            end
            [m(idx), p(idx)] = min(D,[],1);
        end
        
        new_bnd_nodes = m > th^2;
        out.nodes = [out.nodes; M2.nodes(B2(new_bnd_nodes),:)];
        nodemap(B2(new_bnd_nodes)) = max(M1_num_nodes, max(nodemap)) + (1:uint32(nnz(new_bnd_nodes)));
        
        B2(new_bnd_nodes) = [];
        p(new_bnd_nodes) = [];
        
        nodemap(B2) = B1(p); 
    end
    
    M2.elems    = nodemap(M2.elems);
    M2.boundary = nodemap(M2.boundary);
    if isfield(M2, 'electrode')
        for e = 1:numel(M2.electrode)
            try
            M2.electrode(e).nodes = nodemap(M2.electrode(e).nodes);
            end
            try
            M2.electrode(e).faces = nodemap(M2.electrode(e).faces);
            end
        end
    end
    LE = length(out.elems);
    out.elems = [out.elems; M2.elems];
    d1 = size(out.boundary, 2);
    d2 = size(M2.boundary, 2);
    if d1 < d2 % out is 2D or surface mesh
       out.boundary = out.elems;
    end
    if d2 < d1
       M2.boundary = M2.elems;
    end
    out.boundary = unique([out.boundary; M2.boundary], 'rows');
    if isfield(M2, 'electrode') && isstruct(M2.electrode)
        if ~isfield(out, 'electrode'), out.electrode = []; end
        n = numel(out.electrode);
        % is there a better way to do this?
        fn = fieldnames(M2.electrode);
        for j = 1:numel(M2.electrode)
            for f = 1:numel(fn)
                out.electrode(n+j).(fn{f}) =  M2.electrode(j).(fn{f});
            end
        end
    end
    for j = 1:numel(M2.mat_idx)
        out.mat_idx{end+1} = LE+M2.mat_idx{j};
    end
    progress_msg(Inf);
end

% standard field order
out = eidors_obj('fwd_model', out);

function use = nodes_in_bounding_box(LIMnodes,Mnodes,th)
   % limit to nodes in M1 that are within the bounding box of M2
   maxLIM = max(LIMnodes)+th;
   minLIM = min(LIMnodes)-th;
   use = true(length(Mnodes),1);
   for i = 1:size(Mnodes,2)
      use = use & Mnodes(:,i) < maxLIM(i) & Mnodes(:,i) > minLIM(i);
   end

function run_unit_test
subplot(221)
cyl = ng_mk_cyl_models(3,[0],[]);
show_fem(cyl)

subplot(222)
top_nodes = cyl.nodes(:,3)>=1.5;
top_elems = sum(top_nodes(cyl.elems),2)==4;
top.elems = cyl.elems(top_elems,:);
nds = unique(top.elems);
map = zeros(1,length(cyl.nodes));
map(nds) = 1:length(nds);
top.elems = map(top.elems);
top.nodes = cyl.nodes(nds,:);
top.type = 'fwd_model';
top.boundary = find_boundary(top);
show_fem(top)
zlim([0 3]);

subplot(223)
bot_elems = ~top_elems;
bot.elems = cyl.elems(bot_elems,:);
nds = unique(bot.elems);
map = zeros(1,length(cyl.nodes));
map(nds) = 1:length(nds);
bot.elems = map(bot.elems);
bot.nodes = cyl.nodes(nds,:);
bot.type = 'fwd_model';
bot.boundary = find_boundary(bot);
show_fem(bot)
zlim([0 3]);


subplot(224)
M = merge_meshes(bot, top);
show_fem(M);

unit_test_cmp('Number of nodes',length(cyl.nodes), length(M.nodes),0);
unit_test_cmp('Number of elems',length(cyl.elems), length(M.elems),0);