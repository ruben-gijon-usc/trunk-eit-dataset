function [ROIs, imgR, imdl] = define_ROIs(imdl, varargin)
% DEFINE_ROIS: define ROI regions in image
%
% [ROIs, imgR, imdl] = define_ROIs(imdl, xcuts, ycuts, ..., opts)
%   imdl: inv_ or fwd_model structure
%   xcuts, ycuts: x and y cuts of ROI grid
%   opts: options structure
%     opts.reorder = reorder defined ROIs
%     opts.normalize = set ROI of each =1
%     opts.usec2f    = true if use fwd_model.coarse2fine
% 
%   ROIs: matrix [n_ROIs x n_elems]
%   imgR: image of ROIs (mostly for debugging)
%   imdl: inv_model which reconstructs onto ROIs (if possible)
%
% Example:
%  fmdl = mk_library_model('neonate_16el_lungs');
%  fmdl.stimulation = mk_stim_patterns(16,1,[0,1],[0,1],{},1);
%  imdl= select_imdl(fmdl,'GREIT:NF=0.500 32x32');
%  opts.reorder = reshape(1:16,4,4)';
%  [ROIs,imgR] = define_ROIs(imdl,-1:0.5:1,1:-0.5:-1,opts);

% TODO: 
%   - create imdl
%   - add 3D
%   - create rec_model if not there
%   - can it work for non-solve_use_matrix?
%

if ischar(imdl) && strcmp(imdl,'UNIT_TEST'); do_unit_test; return; end

if isstruct(varargin{end})
   opts = varargin{end};
   last = 1;
else
   opts = struct([]);
   last = 0;
end
% Set defaults
%     opts.reorder = reorder defined ROIs
%     opts.normalize = set ROI of each =1
%     opts.usec2f    = true if use fwd_model.coarse2fine
opts = mergestructs(struct( ...
    'reorder',[],'normalize',false,'usec2f',false), opts);

recc2f = 1;
switch imdl.type
   case 'fwd_model';
      fmdl = imdl;
   case 'inv_model';
      if isfield(imdl,'rec_model');
         fmdl = imdl.rec_model;
         recc2f = fmdl.coarse2fine;
      else
         fmdl = imdl.fwd_model;
      end

   otherwise;
      error('define_ROIs: require fwd_ or inv_model');
end
if opts.usec2f
  recc2f = fmdl.coarse2fine;
end


rmdl = mk_grid_model([],varargin{1:end-last});

ROIa = mk_approx_c2f(fmdl,rmdl);
% Octave ./ doesn't expand dimensions for sparse for v9.2
% ROIs = ROIa.' * recc2f./sum(recc2f,1);
% Bug report submitted to octave: 2024-12-19
ROIs = ROIa.' * bsxfun(@rdivide, recc2f, sum(recc2f,1) );
if ~isempty(opts.reorder)
   ROIs = ROIs(opts.reorder(:),:);
end
if opts.normalize
   ROIs = ROIs ./ sum(ROIs,2);
end
imgR= mk_image(fmdl,ROIs.');
imgR.calc_colours.ref_level = 0;

% Can only replace in this case
if strcmp(imdl.type,'inv_model') && ...
   isfield(imdl,'rec_model') && ...
   isfield(imdl,'solve_use_matrix')

   imdl.rec_model.coarse2fine = ROIa;
   imdl.solve_use_matrix.RM = ROIs*imdl.solve_use_matrix.RM;
end

function do_unit_test
   do_GREIT_test
%  do_simple_test
%  do_3D_test

function do_GREIT_test
   imdl = mk_common_model('n3r2',[16,2]);
   [ROIs,imgR] = define_ROIs(imdl,-1:0.5:1,-1:0.5:1);
   unit_test_cmp('n3r2: 2D 1',     size(ROIs), [16,828]);
   unit_test_cmp('n3r2: 2D 2',     ROIs(:,1:9), ...
       [0;0;0;0;0;0;0;0;0;0;0;1;0;0;0;0]*ones(1,9));
   unit_test_cmp('n3r2: 2D 2',     ROIs(:,10:11), ...
       [0,0;0,0;0,0;0,0;0,0;0,0;0,0;0,0;
        0,0;0,0;0,0;95,80;0,0;0,0;0,0;5,20]/100,1e-14);
   [ROIs,imgR] = define_ROIs(imdl,-1:1,-1:1,0:2);
   unit_test_cmp('n3r2: 3D 1',     size(ROIs), [8,828]);
   unit_test_cmp('n3r2: 3D 2',     ROIs(:,1:9), ...
       [0;0;0;1;0;0;0;0]*ones(1,9),1e-14);
   unit_test_cmp('n3r2: 3D 3',     ROIs(:,61:69), ...
       [0;0;1;0;0;0;0;0]*ones(1,9),1e-14);

   fmdl = mk_library_model('neonate_32el_lungs');
   [stim,msel] = mk_stim_patterns(32,1,[0,5],[0,5],{'no_meas_current_next2'},1);
   fmdl.stimulation = stim;
   fmdl.meas_select = msel;
   imdl= select_imdl(fmdl,'GREIT:NF=0.500 32x32');
   [ROIs,imgR] = define_ROIs(imdl,-1:0.5:1,-1:0.5:1);
   subplot(221); show_slices(imgR)
   imdl= select_imdl(fmdl,'GREIT:NF=0.500 16x16');
   [ROIs,imgR] = define_ROIs(imdl,-1:0.5:1,-1:0.5:1);
   subplot(222); show_slices(imgR)
   opts.reorder = reshape(1:16,4,4)';
   [ROIs,imgR] = define_ROIs(imdl,-1:0.5:1,-1:0.5:1,opts);
   subplot(223); show_slices(imgR)
   [ROIs,imgR] = define_ROIs(imdl,-1:0.5:1,1:-0.5:-1,opts);
   subplot(224); show_slices(imgR)
