function fmdl = remove_unused_nodes( fmdl, elec_opt )
% REMOVE_UNUSED_NODES: identify and remove unused nodes in model
% Usage: 
% fmdl = REMOVE_UNUSED_NODES( fmdl );
% fmdl = REMOVE_UNUSED_NODES( fmdl, elec_opt) specifies behavior for empty
% electrodes: 
%   'warn'      - (default) warns about empty electrodes, but keeps them
%   'keep'      - quietly keeps empty electrodes
%   'remove'    - quetly removes empty electrodes
%   'quiet'     - don't warn about changes to electrodes
% 
% See also REMOVE_NODES, REMOVE_ELEMS, CROP_MODEL

% (C) 2019 Andy Adler. License: GPL v2 or v3. $Id: remove_unused_nodes.m 7014 2024-11-27 09:07:51Z bgrychtol $

   if ischar(fmdl) && strcmp(fmdl,'UNIT_TEST'); do_unit_test; return; end

   if nargin < 2 || isempty(elec_opt), elec_opt = 'warn'; end
   quiet = strcmp(elec_opt, 'quiet');
   rm_elec = strcmp(elec_opt, 'remove');
   keep = strcmp(elec_opt, 'keep');

   if num_elems(fmdl)==0; return; end; % don't operate on pathalogical models

   usednodes = unique(fmdl.elems(:));
   if max(usednodes) > num_nodes(fmdl)
      error('remove_unused_nodes: more nodes are used than exist');
   end
   fmdl0 = fmdl;
   nidx = zeros(num_nodes(fmdl),1);
   nidx(usednodes) = 1:length(usednodes);
   fmdl.nodes(nidx==0,:) = [];
   fmdl.elems = remap(nidx, fmdl.elems);

   elecs_changed = false([1 num_elecs(fmdl)]);
   elecs_removed = false([1 num_elecs(fmdl)]); % all nodes/faces removed
   for i=1:num_elecs(fmdl)
      fmdl.electrode(i).nodes =  remap(nidx, fmdl.electrode(i).nodes);
      removed = fmdl.electrode(i).nodes == 0;
      elecs_changed(i) = any(removed);
      fmdl.electrode(i).nodes( removed ) = [];
      % if we had many nodes and less than 3 are left, delete them
      if numel(fmdl.electrode(i).nodes) < 3 && any(removed)
         fmdl.electrode(i).nodes = [];
      end
      elec_empty = isempty(fmdl.electrode(i).nodes);
      if isfield(fmdl.electrode(i),'faces')
          fmdl.electrode(i).faces =  remap(nidx, fmdl.electrode(i).faces);
          removed = any(fmdl.electrode(i).faces== 0,2);
          fmdl.electrode(i).faces(removed,:) = [];
          elec_empty= elec_empty && isempty(fmdl.electrode(i).faces);
      end
      if elec_empty
         elecs_removed(i) = true;
         if eidors_debug('query', 'remove_unused_nodes')
            eidors_msg('Zeros in electrode #%d nodes and faces. Often this means parts of the model are disconnected.',i,1);
            keyboard
         end
      end

   end
   if any(elecs_removed)
      if rm_elec
         fmdl.electrode(elecs_removed) = [];
         eidors_msg('@@ removed electrode(s) # %s', num2str(find(elecs_removed)),1);
      else
         if ~keep && ~quiet
            warning('EIDORS:ElecEmpty', ['Empty electrodes(s) # %s.\n' ...
               'Set eidors_debug(''on'', ''remove_unused_electrodes'') to debug or\n' ...
               'set elec_opt to ''keep'' or ''quiet'' to suppress.'], ...
               num2str(find(elecs_removed)))
         end
         if keep
            eidors_msg('@@ created empty electrode(s) # %s', num2str(find(elecs_removed)),1);
         end
      end
      elecs_changed(elecs_removed) = false; % don't warn twice
   end
   if ~quiet && ~keep && any(elecs_changed)
      warning('EIDORS:ElecChange', ...
         'Removed parts of electrode(s) # %s.', num2str(find(elecs_changed)));
   end
%  fmdl.boundary = find_boundary(fmdl);
   if isfield(fmdl, 'boundary')
      fmdl.boundary = remap(nidx, fmdl.boundary);
      fmdl.boundary(any(fmdl.boundary==0,2),:) = [];
   end
   if isfield(fmdl, 'gnd_node')
      fmdl.gnd_node = nidx(fmdl.gnd_node);
      if fmdl.gnd_node == 0 %% New gnd node if missing
         fmdl = assign_new_gnd_node( fmdl );
      end
   end

%  fmdl.elems = reshape(nidx(fmdl.elems),[],size(fmdl.elems,2));
function mat = remap(nidx,mat)
   mat = reshape(nidx(mat),size(mat));

function fmdl = assign_new_gnd_node( fmdl );
   eidors_msg('FEM_ELECTRODE: Lost ground node => replacing',1);
   d2 = sum((fmdl.nodes - repmat(mean(fmdl.nodes, 1), size(fmdl.nodes, 1), 1)).^2,2);
   [~,fmdl.gnd_node] = min(d2);

function do_unit_test
   fmdl = getfield(mk_common_model('a2c2',4),'fwd_model');
   fmdl.elems(1:4,:) = [];
   fmdl = remove_unused_nodes(fmdl);

   unit_test_cmp('nodes',size(fmdl.nodes),[40,2]);
   unit_test_cmp('elems',size(fmdl.elems),[60,3]);
   unit_test_cmp('ground',fmdl.gnd_node,3);
   
   idx = any(fmdl.elems == fmdl.electrode(1).nodes,2);
   fmdl.elems(idx,:) = []; % remove all elements using that node
   warning('off','EIDORS:ElecEmpty')
   fmdl = remove_unused_nodes(fmdl);
   [~, id] = lastwarn;
   unit_test_cmp('empty warn', id, 'EIDORS:ElecEmpty')
   warning('on', 'EIDORS:ElecEmpty')
   
   fmdl = getfield(mk_common_model('n3r2',[16,2]), 'fwd_model');
   idx = any(fmdl.elems == fmdl.electrode(1).nodes(1),2);
   fmdl.elems(idx,:) = []; % remove all elements using that node
   warning('off','EIDORS:ElecChange')
   fmdl = remove_unused_nodes(fmdl);
   [~, id] = lastwarn;
   unit_test_cmp('change warn', id, 'EIDORS:ElecChange')
   warning('on', 'EIDORS:ElecChange')
