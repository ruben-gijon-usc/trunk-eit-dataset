function imdl = map_RM_slice(imdl,varargin)
%map_RM_slice - map a 3D reconstruction matrix to an arbitrary 2D slice
%
% IMDL2 = map_RM_slice(IMDL3, LEVEL)
% IMDL2 = map_RM_slice(IMDL3, LEVEL, OPT) 
%  Maps the reconstruction matrix in IMDL3.solve_use_matrix.RM onto an 
%  arbitrary slice defined by LEVEL. IMDL3 must be a dual-model inverse 
%  model providing IMDL3.rec_model and IMDL3.fwd_model.coarse2fine.
%  The OPT struct defines the pixel grid of the new reconstruction; see 
%  MK_PIXEL_SLICE for details.
%
% Note: 
% - If the provided imdl.rec_model used medical orientation (decreasing
%   imdl.rec_model.mdl_slice_mapper.y_pts), it may be necessary to flip the
%   orientation of the resultant imdl.rec_model.
% - For selecting a horizontal cut, consider SELECT_RM_SLICE instead. 
% - At the moment, MAP_RM_SLICE only works for voxel-based rec_models as
%   as provided by MK_GRID_MODEL or MK_VOXEL_VOLUME.
%
% See also MK_PIXEL_SLICE, SELECT_RM_SLICE, MK_VOXEL_VOLUME, MK_GRID_MODEL

% TODO:
% - Expand support for any 3D rec_model (see 3D-3D dual tutorial), test
% with inv_solve_diff_GN_one_step (factor out get_RM).


if ischar(imdl) && strcmp(imdl,'UNIT_TEST'); do_unit_test; return; end
[rec, fwd] = mk_pixel_slice(imdl.rec_model,varargin{:});

% f = fwd_model elems
% r = rec_model elems
% c = rec_model values
% s = slice values

c2f = imdl.fwd_model.coarse2fine; % fraction of fwd_model elem in coarse value
c2r = imdl.rec_model.coarse2fine; % fraction of rec_model elem in coarse value
s2r = fwd.coarse2fine;            % fraction of rec_model elem in slice value

rvol = get_elem_volume(imdl.rec_model,'no_c2f');
r2s = s2r' * spdiag(rvol); % volume of each slice elem built from rec elems

% we need to divide by the volume of each pixel, but we don't know
% the height, so we estimate from the area and the maximum volume
s_area = get_elem_volume(rec); 
s_hght = max(sum(r2s,2)) ./ s_area; 
svol = s_area .* s_hght;
% Need bsxfun in octave for sparse divide
% r2s = r2s ./ svol;
r2s = bsxfun(@rdivide, r2s, svol);

if 0
   c = ones(size(c2f,2),1);
   r = c2r * c;
   s = r2s * r;
   unit_test_cmp('still 1', s(30),1,1e-13)
end

s2c = r2s * c2r;
RM = s2c * imdl.solve_use_matrix.RM;

% to map from the slice to the original fwd_model, we need r2c.
% Same approach:
r2c = c2r' * spdiag(rvol);
% again, in general, we don't know the volume of each element (because the
% coarse elements on the rec_model could theoretically extend beyond the 
% elements of it, though we don't build such rec_models). We take the max
% of two estimates:
vol_fmdl = get_elem_volume(imdl.fwd_model);
vol_rmdl = get_elem_volume(imdl.rec_model);
rvol = max([vol_rmdl, vol_fmdl],[],2);
% Need bsxfun in octave for sparse divide
% r2c = r2c ./ rvol;
r2c = bsxfun(@rdivide, r2c, rvol);

c2f = imdl.fwd_model.coarse2fine * r2c * s2r;

if 0
   fmdl = imdl.fwd_model;
   fmdl.coarse2fine = c2f;
   img = mk_image(fmdl, ones(size(c2f,2),1));
   show_fem(img)
end

imdl.fwd_model.coarse2fine = c2f;
imdl.solve_use_matrix.RM = RM;
imdl.rec_model = rec;

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


   img = inv_solve(imdl, vh, vi);
   unit_test_cmp('img size', size(img.elem_data), [4280,5]);

   img.show_slices.img_cols = 1;
   slice1 = calc_slices(img,[inf inf 2]);

   subplot(231); show_fem(fmdl); title 'fmdl'
   img0 = img;
   img0.elem_data = img0.elem_data(:,3);
   subplot(234); show_fem(img0);  title '3D'
   subplot(232); show_slices(img,[inf,inf,2]); title '3D rec';
   eidors_colourbar(img);

   imd2 = map_RM_slice(imdl,[Inf Inf 2.0],struct('z_depth',0.25));
   % flip for medical orientation
   imd2.rec_model.mdl_slice_mapper.y_pts = ...
      fliplr(imd2.rec_model.mdl_slice_mapper.y_pts);

   unit_test_cmp('RM size', size(imd2.solve_use_matrix.RM), [856,928]);
   rm1 = imdl.solve_use_matrix.RM(856*2 + (1:856),:);
   rm2 = imd2.solve_use_matrix.RM;
   unit_test_cmp('RM values', rm2, rm1, 1e-10);

   img = inv_solve(imd2, vh, vi);
   unit_test_cmp('img size', size(img.elem_data), [856,5]);
   im1 = img0.elem_data(856*2 + (1:856));
   unit_test_cmp ('img values', img.elem_data(:,3), im1, 1e-13);

   img.show_slices.img_cols = 1;
   slice2 = calc_slices(img);
   subplot(235), show_slices(img); title('2D rec')
   eidors_colourbar(img);

   subplot(2,3,[3,6])

   level = struct( 'centre', [0 0 2],...
                   'normal_angle',[1 -1 1 0]/sqrt(3)');
   opt.z_depth = 0.5;
   opt.square_pixes = true;
   opt.imgsz = [64 64];
   imd2 = map_RM_slice(imdl, level, opt);
   img = inv_solve(imd2, vh, vi);
   img.elem_data = img.elem_data(:,3);
   h1 = show_fem(img);
%    h1.EdgeColor = 0.8*ones(3,1);

   

   xlim([-1 1]); ylim([-1 1]); zlim([0 4]);
   hold on
   h2 = show_fem(imd2.fwd_model);
   set(h2,'EdgeColor',  0.8*ones(3,1));
   show_fem(img0.fwd_model);
   hold off
   view(-20,40)



   unit_test_cmp('slice Nan', isnan(slice1), isnan(slice2))
   slice1(isnan(slice1))= 0;
   slice2(isnan(slice2))= 0;
   % special case:
   unit_test_cmp('slice values', slice1, slice2, 1e-13)

  

   warning('on','EIDORS:FirstImageOnly');

function fmdl = mystim_square(fmdl);
   [~,fmdl] = elec_rearrange([16,2],'square', fmdl);
   [fmdl.stimulation, fmdl.meas_select] =  ...
       mk_stim_patterns(32,1,[0,5],[0,5],{},1);

function [vh,vi] = test_fwd_solutions;
   posns= linspace(1.0,3.0,5);
   str=''; for i=1:length(posns);
      extra{i} = sprintf('ball%03d',round(posns(i)*100));
      str = [str,sprintf('solid %s=sphere(0.5,0.3,%f;0.1); ', extra{i}, posns(i))];
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
