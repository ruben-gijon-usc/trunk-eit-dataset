function imdl = select_RM_slice(imdl, sel_fcn)
% SELECT_RM_SLICE: cut slices out of an inverse model with a 
%  reconstruction matrix solving onto a 3D (voxel) grid
%
% NOTE that the 3D grid must be of the structure created by MK_GRID_MODEL
% or MK_VOXEL_VOLUME.
%
% Basic usage
% For example, given a 3D GREIT model
%    vopt.zvec= 0.5:0.2:2.5;
%    vopt.imgsz = [32 32];
%    [imdl,opt.distr] = GREIT3D_distribution(fmdl, vopt);
%    imdl = mk_GREIT_model(imdl, 0.20, [], opt);
%
% One could cut the z=1 slice as
%    imdl2= select_RM_slice(imdl, 1.0);
% Or using 
%    sel_fcn = @(z) abs(z-1.0)<0.2; OR
%    sel_fcn = 'abs(z-1.0)<0.2 | abs(z-2.0)<0.2' ; % two planes
%    imdl2= select_RM_slice(imdl, sel_fcn);
%
% SELECT_RM_SLICE applies the selection criteria to element centres. If no
% elements are captured, increase vert_tol (see below).
% 
% Note that the reconstruction planes are between the
% cut planes specified in vopt.zvec
%
% options:
%  imdl.select_RM_slice.vert_tol = .001;
%       % Vertical tolerance for elems on plane (Default .001)
%  imdl.select_RM_slice.keep3D = 1; %keep 3D aspect of cuts
%
% See also MK_GRID_MODEL, MK_VOXEL_VOLUME, SELECT_RM_SLICE

% (C) 2015-2024 Andy Adler and Bartlomiej Grychtol
% License: GPL version 2 or version 3
% $Id: solve_RM_2Dslice.m 7087 2024-12-20 17:44:41Z aadler $

warning('EIDORS:deprecated','SELECT_RM_SLICE is deprecated as of 01-Dec-2024. Use SELECT_RM_SLICE instead.');

if ischar(imdl) && strcmp(imdl,'UNIT_TEST'); do_unit_test; return; end

% Options
vert_tol = .001; try
  vert_tol = imdl.select_RM_slice.vert_tol;
end

keep3D=0; try
  keep3D = imdl.select_RM_slice.keep3D;
end

ctr = interp_mesh(imdl.rec_model); ct3= ctr(:,3);

if isnumeric( sel_fcn )
   ff = abs(ct3-sel_fcn) < vert_tol;
else
    if ischar(sel_fcn) %  we have a string, create a function
      sel_fcn = inline(sel_fcn, 'z');
    end
    ff = feval(sel_fcn,ct3);
end

% mk_grid_model creates 6 tets per voxel. We want to capture the entire
% voxel.
c2f = imdl.rec_model.coarse2fine;
vox_frac = c2f' * ff / 6; % how much of each voxel got captured
ff = c2f * vox_frac>0; % now capture the entire voxel

if ~any(ff)
   error('Found no elements. Check your input or increase opt.vert_tol.')
end

imdl.rec_model.coarse2fine(~ff,:) = [];
imdl.rec_model.elems(~ff,:) = [];

fb = any(imdl.rec_model.coarse2fine,1);
imdl.rec_model.coarse2fine(:,~fb) = [];
imdl.fwd_model.coarse2fine(:,~fb) = [];
imdl.solve_use_matrix.RM(~fb,:) = [];
if isfield(imdl.solve_use_matrix,'PJt')
   imdl.solve_use_matrix.PJt(~fb,:) = [];
end

if keep3D; 
   imdl.rec_model.boundary = find_boundary(imdl.rec_model);
   return;
end

% convert to 2D (only makes sense for a single slice)
imdl.rec_model.nodes(:,3) = [];
keep = false(size(imdl.rec_model.elems,1),1);
keep(1:6:end) = true; keep(6:6:end) = true; % intimate knowledge of mk_grid_model
imdl.rec_model.elems = imdl.rec_model.elems(keep,:);
imdl.rec_model.elems(:,4) = [];
imdl.rec_model.coarse2fine = imdl.rec_model.coarse2fine(keep,:);

nelems = imdl.rec_model.elems;
nnodes = unique(nelems(:));
nnidx = zeros(max(nnodes),1);
nnidx(nnodes) = 1:length(nnodes);
nnodes = imdl.rec_model.nodes(nnodes,:);
nelems = reshape(nnidx(nelems),size(nelems));
imdl.rec_model.nodes = nnodes;
imdl.rec_model.elems = nelems;
imdl.rec_model.boundary = find_boundary(imdl.rec_model);

function do_unit_test
   warning('off','EIDORS:FirstImageOnly');
   [vh,vi] = test_fwd_solutions;
   % inverse geometry
   fmdl= ng_mk_cyl_models([4,1,.5],[16,1.5,2.5],[0.05]);
   fmdl= mystim_square(fmdl);

   vopt.imgsz = [32 32];
   vopt.zvec = linspace( 0.75,3.25,6);
   vopt.save_memory = 1;
%  opt.noise_figure = 2;
   weight = 8.42412109211;
   [imdl,opt.distr] = GREIT3D_distribution(fmdl, vopt);
   imdl = mk_GREIT_model(imdl, 0.2, weight, opt);

   unit_test_cmp('RM size', size(imdl.solve_use_matrix.RM), [4280,928]);
%    unit_test_cmp('RM', imdl.solve_use_matrix.RM(1,1:2), ...
%        [7.896475314622105 -3.130412179150207], 1e-8);



   img = inv_solve(imdl, vh, vi);
   unit_test_cmp('img size', size(img.elem_data), [4280,5]);
   [mm,ll] =max(img.elem_data(:,1));
%    unit_test_cmp('img', [mm,ll], ...
%        [0.426631207850641, 1274], 1e-10);

   img.show_slices.img_cols = 1;
   slice1 = calc_slices(img,[inf inf 2]);
   subplot(231); show_fem(fmdl); title 'fmdl'
   subplot(234); show_fem(img);  title '3D'
   subplot(232); show_slices(img,[inf,inf,2]); title '3D slice';
   got_error = false;
   try 
      imd2 = select_RM_slice(imdl, 2.2);
   catch 
      got_error = true;
   end
   unit_test_cmp('No elem error', got_error, true);
   
   imd2 = select_RM_slice(imdl, 2.0);

   unit_test_cmp('RM size', size(imd2.solve_use_matrix.RM), [856,928]);
%    unit_test_cmp('RM', imd2.solve_use_matrix.RM(1,1:2), ...
%        [-13.546930204198315   9.664897892799864], 1e-8);

   img = inv_solve(imd2, vh, vi);
   unit_test_cmp('img size', size(img.elem_data), [856,5]);
   [mm,ll] =max(img.elem_data(:,1));
%    unit_test_cmp('img', [mm,ll], ...
%        [0.031608449353163, 453], 1e-10);
   slice2 = calc_slices(img);

   unit_test_cmp('slice Nan', isnan(slice1), isnan(slice2))
   slice1(isnan(slice1))= 0;
   slice2(isnan(slice2))= 0;
   unit_test_cmp('slice value', slice1, slice2, 1e-10)

   
   imdl.select_RM_slice.keep3D = 1;
   imd3 = select_RM_slice(imdl, 1.0);
   img = inv_solve(imd3, vh, vi);
   subplot(235); show_fem(img);
   
   sel_fcn = inline('abs(z-1.0)<0.2','z');
   if exist('OCTAVE_VERSION')
      sel_fcn = @(z) abs(z-1.0)<0.2; 
   end
   imd3 = select_RM_slice(imdl, sel_fcn);

   sel_fcn = 'abs(z-1.0)<0.2 | abs(z-2.0)<0.2' ;
   if exist('OCTAVE_VERSION')
      sel_fcn = @(z) abs(z-1.0)<0.2 | abs(z-2.0)<0.2;
   end
   imd3 = select_RM_slice(imdl, sel_fcn);
   img = inv_solve(imd3, vh, vi);
   subplot(236); show_fem(img);

   warning('on','EIDORS:FirstImageOnly');

function fmdl = mystim_square(fmdl);
   [~,fmdl] = elec_rearrange([16,2],'square', fmdl);
   [fmdl.stimulation, fmdl.meas_select] =  ...
       mk_stim_patterns(32,1,[0,5],[0,5],{},1);

function [vh,vi] = test_fwd_solutions;
   posns= linspace(1.0,3.0,5);
   str=''; for i=1:length(posns);
      extra{i} = sprintf('ball%03d',round(posns(i)*100));
      str = [str,sprintf('solid %s=sphere(0.5,0,%f;0.1); ', extra{i}, posns(i))];
   end
   extra{i+1} = str;
   fmdl= ng_mk_cyl_models([4,1,.2],[16,1.5,2.5],[0.05],extra); 
   fmdl = mystim_square(fmdl);
   
   img= mk_image(fmdl,1);
   vh = fwd_solve(img);
   %vh = add_noise(2000, vh);
   for i=1:length(posns);
      img= mk_image(fmdl,1);
      img.elem_data(fmdl.mat_idx{i+1}) = 2;
      vi{i} = fwd_solve(img);
   end;
   vi = [vi{:}];
