function [fmdl, c2f_idx] = remove_elems(fmdl, idx, elec_opt)
%REMOVE_ELEMS   Remove nodes from a fwd_model.
%
%  fmdl = REMOVE_ELEMS(fmdl, idx) removes fmdl.elems(idx,:)
%     fmdl : EIDORS fwd_model
%     idx  : indices (integer or logical) into fmdl.elems
%
%  fmdl = REMOVE_ELEMS(... , elec_opt) specifies treatment of
%  electrodes:
%   'warn'      - (default) warns about empty electrodes, but keeps them
%   'keep'      - quietly keeps empty electrodes
%   'remove'    - quietly removes empty electrodes
%   'quiet'     - don't warn about changes to electrodes
%
%  [fmdl, c2f_idx] = REMOVE_ELEMS(...) also provides indices of elem_data 
%  to keep in an image.
%
%  REMOVE_ELEMS removes affected edges/faces from fmdl.boundary, but does
%  not recalculate it.
%
% See also: REMOVE_NODES, CROP_MODEL, REMOVE_UNUSED_NODES

% (C) 2024 Bartek Grychtol. License: GPL version 2 or version 3
% $Id: remove_elems.m 7002 2024-11-24 13:11:35Z aadler $

if ischar(fmdl) && strcmp(fmdl,'UNIT_TEST'); do_unit_test; return; end

if nargin < 3, elec_opt = ''; end

remove = false([num_elems(fmdl), 1]);
remove(idx) = true;
c2f_idx = [];

if ~any(remove), return, end
if all(remove)
    error('Cannot remove all elements')
end

fmdl.elems(remove,:) = [];

if isfield(fmdl, 'coarse2fine')
    fmdl.coarse2fine(remove,:) = [];
end

% fmdl.boundary = find_boundary(fmdl);

c2f_idx = find(~remove);
elem_map = zeros(size(remove));
elem_map(~remove) = 1:nnz(~remove);

if isfield(fmdl, 'mat_idx')
    mat_removed = false(size(fmdl.mat_idx));
    for i = 1:numel(fmdl.mat_idx)
        fmdl.mat_idx{i} = elem_map(fmdl.mat_idx{i});
        fmdl.mat_idx{i}(fmdl.mat_idx{i} == 0) = [];
        if isempty(fmdl.mat_idx{i})
           mat_removed(i) = true;
        end
    end
    fmdl.mat_idx(mat_removed) = [];
    if isfield(fmdl, 'mat_names')
       fmdl.mat_names(mat_removed) = [];
    end
end


fmdl = remove_unused_nodes(fmdl, elec_opt);


function do_unit_test
f0= mk_common_model('a2c2',8);
f0= f0.fwd_model;
fr= remove_elems(f0, [7,13,14,21,22,32]);
unit_test_cmp('2D- 1a',fr.nodes(11,:), f0.nodes(12,:));
unit_test_cmp('2D- 1b',fr.elems(20,:),[13,14, 6]);

fr= remove_elems(f0, [38,50,51],'remove');
unit_test_cmp('2D- 2e',num_elecs(fr),7);
unit_test_cmp('2D- 2b',fr.nodes(27,:), f0.nodes(27,:));
unit_test_cmp('2D- 2a',fr.nodes(28,:), f0.nodes(29,:));
unit_test_cmp('2D- 2c',f0.elems(40,:),[20,19,33]);
unit_test_cmp('2D- 2d',fr.elems(40,:),[31,19,18]);

fr= remove_elems(f0, [38,50,51],'keep');
unit_test_cmp('2D- 2e',num_elecs(fr),8);
unit_test_cmp('2D- 2b',fr.nodes(27,:), f0.nodes(27,:));
unit_test_cmp('2D- 2a',fr.nodes(28,:), f0.nodes(29,:));
unit_test_cmp('2D- 2c',f0.elems(40,:),[20,19,33]);
unit_test_cmp('2D- 2d',fr.elems(40,:),[31,19,18]);

