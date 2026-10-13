function [imdl fmdl] = mk_pixel_slice(imdl,level,opt)
%MK_PIXEL_SLICE create a pixel model to reconstruct on
% OUT = MK_PIXEL_SLICE(MDL, LEVEL, OPT) creates a slice of pixels as a
% model to reconstruct on. 
%
% Inputs:
%  MDL   = an EIDORS forward or inverse model structure
%  LEVEL = any definition accepted by LEVEL_MODEL_SLICE for a single slice
%  OPT   = an option structure with the following fields and defaults:
%     opt.imgsz = [32 32];    % dimensions of the pixel grid
%     opt.square_pixels = 0;  % adjust imgsz to get square pixels
%     opt.do_coarse2fine = 1; % calcuate c2f on the forward model
%     opt.z_depth = inf;      % z_depth to use with mk_coarse_fine_mapping
%
% Output depends on the type of model suplied. If MDL is a fwd_model
% structure then OUT is a rec_model. If MDL is an inv_model, then OUT is a
% modified version of it, with the pixel slice in inv_model.rec_model and
% updated inv_model.fwd_model.coarse2fine
% 
% [OUT FMDL] = MK_PIXEL_SLICE(MDL, ...) also returns the forward model
% structure with the coarse2fine field.
%
% See also MK_COARSE_FINE_MAPPING, MK_GRID_MODEL, LEVEL_MODEL_SLICE

% (C) 2013 Bartlomiej Grychtol. License: GPL version 2 or 3
% $Id: mk_pixel_slice.m 7070 2024-12-09 17:21:10Z bgrychtol $

if ischar(imdl) && strcmp(imdl,'UNIT_TEST'),do_unit_test;return;end;

switch(imdl.type)
    case 'inv_model'
        fmdl = imdl.fwd_model;
    case 'fwd_model'
        fmdl = imdl;
    otherwise
        error('An EIDORS inverse or forward model struct required');
end

if nargin < 2
   opt = struct; 
else
   opt.level = level;   
end

opt = parse_opts(fmdl,opt);

[rmdl fmdl] = eidors_cache(@do_pixel_slice,{fmdl, opt},'mk_pixel_slice');

switch imdl.type
   case 'inv_model'
      imdl.rec_model = rmdl;
      imdl.fwd_model = fmdl;
   case 'fwd_model'
      imdl = rmdl;
end

function [rmdl fmdl] = do_pixel_slice(fmdl, opt);
[tmp, ~, ~, ifun] =level_model_slice(fmdl,opt.level);
slc = mdl_slice_mesher(tmp,[inf inf 0]);
slc = slc.fwd_model;
mingrid = min(slc.nodes);
maxgrid = max(slc.nodes);
bnd = find_boundary(slc);
% contour_boundary = order_loop(slc.nodes(unique(bnd),:));

if opt.square_pixels ==1
    mdl_sz = maxgrid - mingrid;
    mdl_AR = mdl_sz(1)/mdl_sz(2);
    img_AR = opt.imgsz(1)/opt.imgsz(2);
    if mdl_AR < img_AR
        delta = (mdl_sz(2) * img_AR - mdl_sz(1)) /2;
        mingrid(1) = mingrid(1) - delta;
        maxgrid(1) = maxgrid(1) + delta;
    elseif mdl_AR > img_AR
        delta = (mdl_sz(1)/img_AR - mdl_sz(2)) / 2;
        mingrid(2) = mingrid(2) - delta;
        maxgrid(2) = maxgrid(2) + delta;
    end
end

xgrid = linspace(mingrid(1),maxgrid(1),opt.imgsz(1)+1);
ygrid = linspace(mingrid(2),maxgrid(2),opt.imgsz(2)+1);
rmdl = mk_grid_model([],xgrid,ygrid);
x_pts = xgrid(1:end-1) + 0.5*diff(xgrid);
y_pts = ygrid(1:end-1) + 0.5*diff(ygrid);
% y_pts = fliplr(y_pts); %medical

% NOTE: This controls the image resolution. If you want higher res, you
% need to either specify it in opt.imgsz or manually overwrite (or remove)
% the imdl.rec_model.mdl_slice_mapper.
rmdl.mdl_slice_mapper.x_pts = x_pts;
rmdl.mdl_slice_mapper.y_pts = y_pts;
rmdl.mdl_slice_mapper.level = opt.level;
rmdl.mdl_slice_mapper.model_2d = 1;
x_avg = conv2(xgrid, [1,1]/2,'valid');
y_avg = conv2(ygrid, [1,1]/2,'valid');
[x,y] = ndgrid( x_avg, y_avg);

% 20141119: The inpolygon approach fails on non-simply-connected domains
% inside = inpolygon(x(:),y(:),contour_boundary(:,1),contour_boundary(:,2) );
P = [x(:) y(:)]; % P(end,3) = 0;
inside = any(point_in_triangle(P, slc.elems, slc.nodes(:,1:2)),2);
% inside = full(inside);

ff = find(~inside);

if opt.do_coarse2fine
%     rmdl.mk_coarse_fine_mapping.z_depth = opt.z_depth;
%     fmdl.coarse2fine = mk_coarse_fine_mapping(tmp,rmdl);
%     fmdl.coarse2fine(:,ff) = [];
    if isinf(opt.z_depth)
       zgrid = [min(tmp.nodes(:,3))-1 max(tmp.nodes(:,3))+1];
    else
       zgrid = [-opt.z_depth opt.z_depth];
    end
    [~, c2f] = mk_grid_model(tmp,xgrid,ygrid,zgrid);
    c2f(:,ff) = [];
    fmdl = eidors_obj('set',fmdl,'coarse2fine', c2f);
end

rmdl.elems([2*ff, 2*ff-1],:)= [];
rmdl.coarse2fine([2*ff, 2*ff-1],:)= [];
rmdl.coarse2fine(:,ff)= [];
% rmdl.boundary = find_boundary(rmdl);
% show individual elements (more like how the 2d grid models display)
rmdl.boundary = rmdl.elems;
rmdl.inside   = inside; % the inside array is useful in other functions


if isfield(fmdl,'mat_idx')
   rmdl.mat_idx = calc_mat_idx(rmdl,fmdl,ff,opt);
end

% map electrodes
rmdl.nodes(:,3) = 0;
rmdl.nodes =  ifun(rmdl.nodes);
slc.nodes =  ifun(slc.nodes);

% no longer needed, level_model_slice does this
% if ~isstruct(opt.level)
%    isf = ~isinf(opt.level);
%    if nnz(isf) == 1
%       rmdl.nodes(:,isf) = opt.level(:,isf);
%    end
% end

if isfield(slc, 'electrode')
   for i = flipud(1:numel(slc.electrode))
        tmp = rmfield(slc.electrode(i), 'nodes');
        x_elec = slc.nodes( [slc.electrode(i).nodes], 1);
        y_elec = slc.nodes( [slc.electrode(i).nodes], 2);
        z_elec = slc.nodes( [slc.electrode(i).nodes], 3);
        tmp.pos       = [x_elec, y_elec, z_elec];
        elec(i) = tmp;
    end
    rmdl.electrode = elec;
end
   
% Not needed any more (uses fwd_model.mdl_slice_mapper)
% rmdl.show_slices.levels = opt.level;

% not quite correct, but 'rec_model' has many side effects
rmdl = eidors_obj('fwd_model', rmdl); 
      

function mat_idx = calc_mat_idx(rmdl,fmdl,ff,opt)
   % calculate mat_idx for the rec_model
   fmdl.mdl_slice_mapper = rmfield(rmdl.mdl_slice_mapper,'model_2d');
   fmdl.mdl_slice_mapper.level = opt.level;
   img = mk_image(fmdl,0);
   for i = 1:length(fmdl.mat_idx);
      img.elem_data(fmdl.mat_idx{i}) = i;
   end
   slice = calc_slices(img); % level specified in fmdl.mdl_slice_mapper
   slice = slice';
   mat = reshape([slice(:)'; slice(:)'],1,[]);
   mat([2*ff, 2*ff-1])= [];
   mat_idx = cell(max(mat),1);
   for i = 1:max(mat)
      mat_idx(i) = {find(mat==i)'};
   end


 function opt = parse_opts(fmdl, opt)
    if ~isfield(opt, 'imgsz')     
        opt.imgsz = [32 32]; 
    end
    if ~isfield(opt, 'square_pixels')
        opt.square_pixels = 0;
    end
    if ~isfield(opt, 'level') || isempty(opt.level)
        opt.level = get_elec_level(fmdl);
    else
        if ~isstruct(opt.level) && numel(opt.level) ==1
            opt.level = [inf inf opt.level];
        end
    end
    if ~isfield(opt, 'do_coarse2fine')
        opt.do_coarse2fine = 1;
    end
    if ~isfield(opt, 'z_depth')
        opt.z_depth = inf;
    end
    
function elec_lev = get_elec_level(fmdl)
    z_elec= fmdl.nodes( [fmdl.electrode(:).nodes], 3);
    min_e = min(z_elec); max_e = max(z_elec);
    elec_lev = [inf,inf,mean([min_e,max_e])];

    
function do_unit_test
    imdl = mk_common_model('n3r2',[16,2]);
    opt.square_pixels = 1;
    opt.imgsz = [16 16];
    mdl = mk_pixel_slice(imdl.fwd_model,[inf 2 2.5], opt);
    img = mk_image(mdl,1);
    
    subplot(231)
    show_fem(imdl.fwd_model);
    view([-50 10])

    subplot(232)
    show_fem(img);
    zlim([0 3]);
    ylim([-1 1])
    xlim([-1 1]);
    view([-50 10])
    
    subplot(233)
    show_slices(img);
    
    subplot(234)
    imdl = mk_pixel_slice(imdl);
    img = mk_image(imdl.rec_model,1);
    show_fem(img);
    zlim([0 3]);
    ylim([-1 1])
    xlim([-1 1]);
    view([-50 10])
    
    subplot(235)
    mdl = mk_library_model('pig_23kg_16el_lungs');
    img = mk_image(mdl,1);
    for i = 1:length(mdl.mat_idx)
       img.elem_data(mdl.mat_idx{i}) = i;
    end
    show_fem(img)
    view(2)
    
    subplot(236)
    clear opt
    opt.imgsz = [64 64];
    opt.square_pixels = 1;
    opt.do_coarse2fine = 0;
    mdl = mk_library_model('pig_23kg_16el_lungs');
%     rmdl = mk_pixel_slice(mdl,opt); % deprecated
    rmdl = mk_pixel_slice(mdl,[],opt);
    img = mk_image(rmdl,1);
    for i = 1:length(rmdl.mat_idx)
       img.elem_data(rmdl.mat_idx{i}) = i;
    end
    show_slices(img);
