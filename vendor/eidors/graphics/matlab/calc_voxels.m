function [V, rimg] = calc_voxels(img, opt, level)
%CALC_VOXELS Calculate volumetric data from an image
% V = CALC_VOXELS(IMG) calculates volumetric representation of the image 
% IMG using a 3D grid with a maximum dimension of 32. V is a 3-D or 4-D
% matrix with dimensions Y x X x Z x N, where N is the number of frames in
% the image data (time), and X, Y, Z refer to the axes of the image node 
% space. 
%
% V = CALC_VOXELS(IMG, OPT) allows specifying an options struct:
%   OPT.method  - specifies the method of determining value in each voxel:
%                   'centre'  - the value at its centre (default)
%                   'average' - the average value (expensive)
%                 The average method can only be used for images with
%                 values specified on elements (elem_data). It produces
%                 smoother images that simulate the partial volume effect 
%                 and look better at lower resolution than the 'centre' 
%                 method.
%   OPT.imgsz   - specifies the target grid size (default [32 32 32]). 
%                 See DEFINE_VOXELS for details, alternatives and 
%                 additional options.
%                 voxels supported by DEFINE_VOXELS can also be used.
%   OPT.save_memory - only useful if method is 'average', reduces memory
%                 consumption, at the expense of time. See MK_GRID_C2F.
%                 All other options of MK_GRID_C2F may also be specified.
% NOTE: The OPT struct is passed in its entirety to both DEFINE_VOXELS and 
%        MK_GRID_C2F. 
%                
% V = CALC_VOXELS(IMG, OPT, LEVEL) specifies the orientaion of the XY plane
% of the voxel grid. Any definition of a single slice accepted by
% LEVEL_MODEL_SLICE can be used. OPT can be an empty struct. 
%
% [V, RIMG] = CALC_VOXELS(_) also returns a grid-based image as produced
% by MK_GRID_MODEL, suitable for use with SHOW_FEM.
%
% CALC_VOXELS supports parametrization through DATA_MAPPER.
%
% Example:
%  fmdl = ng_mk_ellip_models([1,1.5,0.8,0.1],[],[]);
%  fmdl.nodes(:,3) = fmdl.nodes(:,3) - .25;
%  fmdl = fix_model(fmdl, struct('elem_centre',true));
%  img = mk_image(fmdl, prod(fmdl.elem_centre,2));
%  [V, rimg] = calc_voxels(img);
%  subplot(221), show_fem(img)
%  subplot(222), show_fem(rimg);
%  subplot(223), show_slices(rimg,3);
%  subplot(224), show_slices(V);
%
% See also DEFINE_VOXELS, CALC_GRID_POINTS, MK_GRID_C2F, MK_GRID_MODEL, 
% CALC_SLICES, DATA_MAPPER

% (C) 2024 Bartlomiej Grychtol
% License: GPL version 2 or 3
% $Id: calc_voxels.m 7094 2024-12-20 23:05:22Z bgrychtol $

if ischar(img) && strcmp(img, 'UNIT_TEST'), do_unit_test; return, end

if nargin < 2
    opt = struct();
end

if ~isfield(opt, 'method')
    opt.method = 'centre';
end

img = data_mapper(img); % support parametrization

do_rimg = false;

switch opt.method
    case 'centre'
        do_c2f = false;
    case 'average'
        do_rimg = true;
        do_c2f = true;
    otherwise
        error('EIDORS:WrongInput', 'Method must be centre or average.');
end

if isfield(img, 'node_data') && strcmp(opt.method, 'average')
    error('Only the centre method can be used with node data.');
end

if nargout > 1
    do_rimg = true;
end

if nargin > 2
    img.fwd_model = level_model_slice(img.fwd_model, level);
end

[~, ~, ~, opt] = define_voxels(img.fwd_model, opt);

if do_c2f % method 'average'
    rmdl = mk_grid_model([],opt.xvec,opt.yvec,opt.zvec);
    c2f = mk_grid_c2f(img.fwd_model, rmdl, opt);
    fvol = get_elem_volume(img.fwd_model,'no_c2f');
    % need bsxfun because of octave, and it's super slow
    f2c = bsxfun(@times, c2f, fvol)';
    %f2c = f2c ./ sum(f2c,2); % MATLAB:array:SizeLimitExceeded
    f2c = spdiag(1./sum(f2c,2)) * f2c;
    rimg = mk_image(rmdl,f2c*img.elem_data);
    V = calc_grid_points(rimg, opt.x_pts, opt.y_pts, opt.z_pts);
else % method 'centre'
    V = calc_grid_points(img, opt.x_pts, opt.y_pts, opt.z_pts);
    if do_rimg
        rmdl = mk_grid_model([],opt.xvec,opt.yvec,opt.zvec);
        num_elems = size(rmdl.coarse2fine,2);
        rimg = mk_image(rmdl, reshape(V,num_elems,[]));
    end
end

V = permute(V, [2 1 3 4]);

function do_unit_test
do_cases
do_example

function do_cases

imdl= mk_common_model('n3r2',[16,2]);
fmdl = imdl.fwd_model;

obj01 = [390,391,393,396,402,478,479,480,484,486, ...
         664,665,666,667,668,670,671,672,676,677, ...
         678,755,760,761];
obj02 = [318,319,321,324,330,439,440,441,445,447, ...
         592,593,594,595,596,598,599,600,604,605, ...
         606,716,721,722];

test = 1;
while 1
    img = mk_image(fmdl, 1);
    opt = struct;
    error_expected = false;
    switch test
        case 1
            opt.imgsz = [32 32 32];
            opt.cube_voxels = 1;
            img.elem_data(obj01) = 1.15;
            img.elem_data(obj02) = 0.8;
            expect_out =  [22 22 32];
        case 2
            opt.imgsz = [32 32 32];
            opt.cube_voxels = 1;
            img.elem_data(:,2) = img.elem_data;
            img.elem_data(obj01,1) = 1.15;
            img.elem_data(obj02,2) = 0.8;
            expect_out =  [22 22 32 2];
        case 3
            opt.imgsz = [32 32 32];
            opt.cube_voxels = 1;
            img.elem_data(:,2) = img.elem_data;
            img.elem_data(obj01,1) = 1.15;
            img.elem_data(obj02,2) = 0.8;
            opt.method = 'average';
            expect_out =  [22 22 32 2];
        case 4
            opt.imgsz = [32 32 32];
            opt.cube_voxels = 1;
            img = mk_image(fmdl, 1:num_nodes(fmdl));
            expect_out =  [22 22 32];
        case 5
            opt.imgsz = [32 32 32];
            opt.cube_voxels = 1;
            img = mk_image(fmdl, 1:num_nodes(fmdl));
            opt.method = 'average';
            error_expected = true;
        case 6 % empty options
            img.elem_data(obj01) = 1.15;
            img.elem_data(obj02) = 0.8;
            expect_out =  [22 22 32];
        otherwise
            break
    end
    test_txt = sprintf('calc_voxels T#%2d:', test);
    try
        [V, rimg] = calc_voxels(img, opt);
        unit_test_cmp(test_txt, size(V), expect_out);
%       fprintf('Test #%2d: %s\n', test, mat2str(size(V)));
    catch e
        if error_expected
            unit_test_cmp([test_txt, ' expected error'], error_expected,true);
        else
            fprintf('Test #%2d: error\n', test);
        end
    end

    show_test(img, V, rimg)
    test = test + 1;
end


function show_test(img, V, rimg)
s = warning('query','EIDORS:FirstImageOnly');
warning('off','EIDORS:FirstImageOnly');
try
    clf
    subplot(131)
    show_fem(img)
    subplot(132)
    show_fem(rimg)
    subplot(133)
    show_slices(V)
catch e
    warning(s, 'EIDORS:FirstImageOnly');
    rethrow(e)
end

function do_example
 fmdl = ng_mk_ellip_models([1,1.5,0.8,0.1],[],[]);
 fmdl.nodes(:,3) = fmdl.nodes(:,3) - .25;
 fmdl = fix_model(fmdl, struct('elem_centre',true));
 img = mk_image(fmdl, prod(fmdl.elem_centre,2));
 [V, rimg] = calc_voxels(img);
 subplot(221), show_fem(img)
 subplot(222), show_fem(rimg);
 subplot(223), show_slices(rimg,3);
 subplot(224), show_slices(V);
