function [imdl, fmdl] = mk_voxel_volume(varargin)
%MK_VOXEL_VOLUME create a voxel model to reconstruct on
% OUT = MK_VOXEL_VOLUME(MDL)
% OUT = MK_VOXEL_VOLUME(MDL, OPT)
%
% Inputs:
%  MDL   = an EIDORS forward or inverse model structure
%  OPT   = an option structure with the following fields and defaults:
%     opt.imgsz = [32 32 4]; % X, Y and Z dimensions of the voxel grid
%     opt.xvec  = []          % Specific X cut-planes between voxels
%                             % A scalar means the number of planes
%                             % Takes precedence over other options
%     opt.yvec  = []          % Specific Y cut-planes between voxels
%                             % A scalar means the number of planes
%                             % Takes precedence over other options
%     opt.zvec  = []          % Specific Z cut-planes between voxels
%                             % A scalar means the number of planes
%                             % Takes precedence over other options
%     opt.square_pixels = 0;  % adjust imgsz to get square pixels (in XY)
%     opt.cube_voxels = 0;    % adjust imgsz to get cube voxels (in XYZ)  
%     opt.prune_model = true  % removes voxels outside the supplied MDL
%                             % This runs mk_grid_c2f. For simple
%                             % geometries, such a cylinder, it is much
%                             % quicker to set to false and prune manually.
%     opt.save_memory         % passed to mk_grid_c2f
%
% Output depends on the type of model supplied. If MDL is a fwd_model
% structure then OUT is a rec_model. If MDL is an inv_model, then OUT is a
% modified version of it, with the voxel volume in inv_model.rec_model and
% updated inv_model.fwd_model.coarse2fine
% 
% [OUT FMDL] = MK_VOXEL_VOLUME(MDL, ...) also returns the forward model
% structure with the coarse2fine field.
%
% See also MK_PIXEL_SLICE, MK_GRID_MODEL, MK_GRID_C2F

% (C) 2015 Bartlomiej Grychtol - all rights reserved by Swisstom AG
% License: GPL version 2 or 3
% $Id: mk_voxel_volume.m 6809 2024-04-23 14:14:43Z aadler $

% >> SWISSTOM CONTRIBUTION <<

imdl = varargin{1};
% if input is 'UNIT_TEST', run tests
if ischar(imdl) && strcmp(imdl,'UNIT_TEST') 
    do_unit_test; clear imdl 
    return; 
end

if nargin < 2
    opt = struct;
else 
    opt = varargin{2};
end

switch(imdl.type)
    case 'inv_model'
        fmdl = imdl.fwd_model;
    case 'fwd_model'
        fmdl = imdl;
    otherwise
        error('An EIDORS inverse or forward model struct required');
end

opt = parse_opts(fmdl,opt);

copt.fstr = 'mk_voxel_volume';
copt.cache_obj = get_cache_obj(fmdl, opt);
[rmdl, c2f] = eidors_cache(@do_voxel_volume,{fmdl, opt},copt);

if ~isempty(c2f)
    fmdl.coarse2fine = c2f;
end

switch imdl.type
   case 'inv_model'
      imdl.rec_model = rmdl;
      imdl.fwd_model = fmdl;
   case 'fwd_model'
      imdl = rmdl;
end


%-------------------------------------------------------------------------%
% The main function
function [rmdl, c2f] = do_voxel_volume(fmdl,opt)

    rmdl = mk_grid_model([],opt.xvec,opt.yvec,opt.zvec);
    
       
    c2f = [];
    if ~opt.prune_model, return, end
    
    %     fmdl.elems = fmdl.elems( 210714,:);
    [c2f, m]  = mk_grid_c2f(fmdl, rmdl, opt);
    
    inside = any(c2f,1);
    c2f(:,~inside) = [];
    rm = ~logical(rmdl.coarse2fine*inside');
    rmdl.elems(rm,:) = [];
    rmdl.coarse2fine(rm,:) = [];
    rmdl.coarse2fine(:,~inside) = [];
    
    
    bnd_fcs = ones(1,nnz(inside))*m.vox2face(inside,:) == 1;
    rmdl.boundary = m.faces(bnd_fcs,:);
    rmdl.inside   = inside; % the inside array is useful in other functions
    
    rmdl.mdl_slice_mapper.x_pts = opt.x_pts;
    rmdl.mdl_slice_mapper.y_pts = opt.y_pts;
    rmdl.mdl_slice_mapper.z_pts = opt.z_pts;
    
    
%-------------------------------------------------------------------------%
% Assemble a reference object for caching
function cache_obj = get_cache_obj(fmdl, opt)
    tmp = struct;
    flds = {'nodes','elems'};
    for f = flds;
        tmp.(f{1}) = fmdl.(f{1});
    end
    cache_obj = {tmp, opt};
    

%-------------------------------------------------------------------------%
% Parse option struct
 function opt = parse_opts(fmdl, opt)
    if ~isfield(opt, 'imgsz')     
        opt.imgsz = [32 32 4]; 
    end
    if ~isfield(opt, 'square_pixels')
        opt.square_pixels = 0;
    end
    if ~isfield(opt, 'cube_voxels')
        opt.cube_voxels = 0;
    end
    if ~isfield(opt, 'xvec')
        opt.xvec = [];
    end
    if ~isfield(opt, 'yvec')
        opt.yvec = [];
    end
    if ~isfield(opt, 'zvec')
        opt.zvec = [];
    end
    if ~isfield(opt, 'prune_model')
        opt.prune_model = true;
    end
    
    [~,~,~,opt] = define_voxels(fmdl, opt);
    
    if ~isfield(opt, 'do_coarse2fine')
        opt.do_coarse2fine = 1;
    end
    if ~isfield(opt, 'z_depth')
        opt.z_depth = inf;
    end

%-------------------------------------------------------------------------%
% Perfom unit tests
function do_unit_test
   % Check for convhull errors
   vopt.imgsz = [10,10,10];
   imdl = mk_common_model('n3r2',[16,2]);
   [img.fwd_model,cmdl] = mk_voxel_volume(imdl.fwd_model,vopt);
   unit_test_cmp('Simple',cmdl.elems(3,:),[64,65,96,33]);


fmdl = mk_library_model('adult_male_16el');
% fmdl= ng_mk_cyl_models([2,2,.4],[16,1],[.1,0,.025]);
opt.square_pixels = 1;
[rmdl, fmdl] = mk_voxel_volume(fmdl, opt);


subplot(121)
rimg = mk_image(rmdl,0);
rimg.elem_data = zeros(size(rmdl.coarse2fine,2),1);
idx = round(rand(5,1)* length(rimg.elem_data));
rimg.elem_data(idx) = 1;
show_fem(rimg);

subplot(122)
img = mk_image(fmdl,0);
img.elem_data = fmdl.coarse2fine * rimg.elem_data;
img.calc_colours.ref_level = 0;
img.calc_colours.transparency_thresh = 1e-2;

show_fem(img);





