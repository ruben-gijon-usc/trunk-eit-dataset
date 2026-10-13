function [fmdl, c2f_idx] = remove_nodes(fmdl, idx, elec_opt)
%REMOVE_NODES   Remove nodes from a fwd_model.
%
%  fmdl = REMOVE_NODES(fmdl, idx) removes fmdl.nodes(idx,:)
%     fmdl : EIDORS fwd_model
%     idx  : indices (integer or logical) into fmdl.nodes
%
%  fmdl = REMOVE_NODES(... , elec_opt) specifies treatment of
%    electrodes (fmdl.electrode) affected by the node removal:
%     'warn'      - (default) warns about empty electrodes, but keeps them
%     'keep'      - keeps empty electrodes
%     'remove'    - removes empty electrodes
%     'quiet'     - don't warn about changes to electrodes
%    These options have no impact on the actual node removal.
%
%  [fmdl, c2f_idx] = REMOVE_NODES(...) also provides indices of elem_data 
%  to keep in an image.
%
%  REMOVE_NODES removes affected edges/faces from fmdl.boundary, but does
%  not recalculate it.
%
% See also: REMOVE_ELEMS, CROP_MODEL, REMOVE_UNUSED_NODES

% (C) 2024 Bartek Grychtol. License: GPL version 2 or version 3
% $Id: remove_nodes.m 7002 2024-11-24 13:11:35Z aadler $



if ischar(fmdl) && strcmp(fmdl, 'UNIT_TEST'), do_unit_test; return, end
if nargin < 3, elec_opt = ''; end

node_idx = false([num_nodes(fmdl), 1]);
node_idx(idx) = true;
elem_idx = any(node_idx(fmdl.elems),2);

[fmdl, c2f_idx] = remove_elems(fmdl, elem_idx, elec_opt);

function do_unit_test

f0= mk_common_model('a2c2',4);
f0= f0.fwd_model;
fr= remove_nodes(f0, [10:13]);
unit_test_cmp('2D- 1a',fr.nodes(20,:), f0.nodes(24,:));
unit_test_cmp('2D- 1b',fr.elems(20,:),[24,11,12]);

fr= remove_nodes(f0, [26:33],'remove');
unit_test_cmp('2D- 2a',fr.nodes(25,:), f0.nodes(25,:));
unit_test_cmp('2D- 2b',fr.nodes(26,:), f0.nodes(34,:));
unit_test_cmp('2D- 2c',f0.elems(40,:),[20,19,33]);
unit_test_cmp('2D- 2d',fr.elems(40,:),[14,25,33]);
unit_test_cmp('2D- 2e',num_elecs(fr),2);

fr= remove_nodes(f0, [26:33],'keep');
unit_test_cmp('2D- 3a',fr.nodes(25,:), f0.nodes(25,:));
unit_test_cmp('2D- 3b',fr.nodes(26,:), f0.nodes(34,:));
unit_test_cmp('2D- 3c',f0.elems(40,:),[20,19,33]);
unit_test_cmp('2D- 3d',fr.elems(40,:),[14,25,33]);
unit_test_cmp('2D- 3e',num_elecs(fr),4);

f0= mk_common_model('n3r2',[16,2]);
f0= f0.fwd_model;
fr= remove_nodes(f0, [59:63]);
unit_test_cmp('3D- 1a',fr.nodes(80,:), f0.nodes(85,:));
unit_test_cmp('3D- 1b',fr.elems(80,:),[41,42,76,18]);

fr= remove_nodes(f0, [22:27],'remove');
unit_test_cmp('2D- 2a',fr.nodes(21,:), f0.nodes(21,:));
unit_test_cmp('2D- 2b',fr.nodes(22,:), f0.nodes(28,:));
unit_test_cmp('2D- 2c',f0.elems(40,:),[10,73,74,38]);
unit_test_cmp('2D- 2d',fr.elems(40,:),[10,67,68,32]);
unit_test_cmp('2D- 2e',num_elecs(fr),29);

fr= remove_nodes(f0, [22:27],'keep');
unit_test_cmp('2D- 3a',fr.nodes(21,:), f0.nodes(21,:));
unit_test_cmp('2D- 3b',fr.nodes(22,:), f0.nodes(28,:));
unit_test_cmp('2D- 3c',f0.elems(40,:),[10,73,74,38]);
unit_test_cmp('2D- 3d',fr.elems(40,:),[10,67,68,32]);
unit_test_cmp('2D- 3e',num_elecs(fr),32);

