function [xvec, yvec, zvec, opt] = define_voxels(fmdl, varargin)
%DEFINE VOXELS define edges of a voxel grid covering a model 
% [XVEC, YVEC, ZVEC, OPT] = DEFINE_VOXELS(FMDL)
% [XVEC, YVEC, ZVEC, OPT] = DEFINE_VOXELS(FMDL, OPT)
%
% Inputs:
%  FMDL  = an EIDORS forward model structure
%  OPT   = an option structure with the following fields and defaults:
%     opt.imgsz = [32 32 32]; % X, Y and Z dimensions of the voxel grid.
%                             % If opt.zvec is provided, Z need not be 
%                             % specified. If imgsz is a scalar, X = Y = Z
%                             % and cube_voxels defaults to 1;
%     opt.xvec  = []          % Specific X cut-planes between voxels
%                             % A scalar means the number of planes
%                             % Takes precedence over other options
%     opt.yvec  = []          % Specific Y cut-planes between voxels
%                             % A scalar means the number of planes
%                             % Takes precedence over other options
%     opt.zvec  = []          % Specific Z cut-planes between voxels
%                             % A scalar means the number of planes
%                             % Takes precedence over other options
%     opt.xlim  = []          % 2-element vector specifying the minimum and
%                             % maximum extent in the X direction. 
%                             % Default: extent of FMDL.nodes
%     opt.ylim  = []          % 2-element vector specifying the minimum and
%                             % maximum extent in the X direction. 
%                             % Default: extent of FMDL.nodes
%     opt.zlim  = []          % 2-element vector specifying the minimum and
%                             % maximum extent in the X direction. 
%                             % Default: extent of FMDL.nodes
%     opt.square_pixels = 1;  % adjust imgsz to get square pixels (in XY)
%                             % Default: 1 if imgsz is scalar, 0 otherwise
%     opt.cube_voxels = 1;    % adjust imgsz to get cube voxels (in XYZ)  
%                             % Default: 1 if imgsz is scalar, 0 otherwise
%
% Outputs:
%   XVEC, YVEC, ZVEC = edges of voxels suitable for use with NDGRID
%   OPT              = an option structure as above with all fields present 
%                      and consistent, and the additional fields 
%                      x_pts, y_pts, and z_pts defining cordinates of voxel
%                      centers (to be passed to mdl_slice_mapper)
%
% DEFINE_VOXELS(FMDL, 'Key', Value, ...) is an alternative way to specify
% options.
%
% Example:
%   [xvec, yvec, zvec] = define_voxels(img.fwd_model);
%   [~,~,~,opt] = define_voxels(fmdl,'imgsz',32,'zvec',[0,1,2,3]);
%   [~,~,~,opt] = define_voxels(fmdl,'imgsz',32, ...
%                                    'cube_voxels', 0, 'square_pixels', 0);
% See also MK_VOXEL_VOLUME, CALC_VOXELS

% (C) 2015-2024 Bartlomiej Grychtol - all rights reserved by Swisstom AG
% License: GPL version 2 or 3
% $Id: define_voxels.m 6877 2024-05-10 13:26:26Z bgrychtol $

% >> SWISSTOM CONTRIBUTION <<

switch fmdl.type
   case 'fwd_model'
      %nothing
   case 'image'
      fmdl = fmdl.fwd_model;
   otherwise
      error('EIDORS:WrongInput', 'Expected EIDORS image or fwd_model.')
end

if nargin < 2
    opt = struct;
elseif ischar(varargin{1})
    opt = struct(varargin{:});
else
    opt = varargin{1};
end

if ~isfield(opt, 'imgsz')
    opt.imgsz = 32;
end
fields = {'xvec', 'yvec', 'zvec', 'xlim', 'ylim', 'zlim'};
for f = 1: numel(fields)
   if ~isfield(opt, fields{f})
      opt.(fields{f}) = [];
   end
end
if isempty(opt.xvec) && isempty(opt.imgsz)
    error('EIDORS:WrongInput', 'opt.imgsz must not be empty if opt.xvec is empty or absent');
end
if isscalar(opt.imgsz)
    opt.imgsz(2:3) = opt.imgsz;
    if ~isfield(opt, 'square_pixels')
        opt.square_pixels = 1;
    end
    if ~isfield(opt, 'cube_voxels')
        opt.cube_voxels = 1;
    end
end
if isempty(opt.zvec) && numel(opt.imgsz) == 2
    error('EIDORS:WrongInput', 'opt.imgsz must have 3 elements if opt.zvec is empty or absent');
end

if ~isfield(opt, 'square_pixels')
    opt.square_pixels = 0;
end
if ~isfield(opt, 'cube_voxels')
    opt.cube_voxels = 0;
end

mingrid = min(fmdl.nodes);
maxgrid = max(fmdl.nodes);
fields = {'xlim', 'ylim', 'zlim'};
for f = 1:3
   if ~isempty(opt.(fields{f}))
      mingrid(f) = opt.(fields{f})(1);
      maxgrid(f) = opt.(fields{f})(2);
   else
      opt.(fields{f})(1) = mingrid(f);
      opt.(fields{f})(2) = maxgrid(f);
   end
end

mdl_sz = maxgrid - mingrid;

allempty = isempty(opt.xvec) & isempty(opt.yvec) & isempty(opt.zvec);
if opt.cube_voxels && ~allempty
    warning('EIDORS:IncompatibleOptions','Option cube_voxels is set to 0 when xvec, yvec or zvec is specifed');
    opt.cube_voxels = 0;
end
if opt.cube_voxels && allempty
    side_sz = max(mdl_sz ./ opt.imgsz(1:3));
    n_steps = ceil(mdl_sz / side_sz);
    mdl_ctr = mingrid + mdl_sz/2;
    mingrid = mdl_ctr - n_steps/2 * side_sz;
    maxgrid = mdl_ctr + n_steps/2 * side_sz;
    opt.imgsz = n_steps;
    
elseif opt.square_pixels
    if ~isempty(opt.xvec) || ~isempty(opt.yvec)
        warning('EIDORS:IncompatibleOptions','Option square_pixels is set to 0 when xvec or yvec is specifed');
        opt.square_pixels = 0;
    else
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
end

fields = {'xvec', 'yvec', 'zvec'};
pts_fields = {'x_pts', 'y_pts', 'z_pts'};
for i = 1:3
    if isscalar(opt.(fields{i}))
        opt.imgsz(i) = opt.(fields{i}); % make imgsz consistent
        opt.(fields{i}) = [];
    end
    if isempty(opt.(fields{i}))
        opt.(fields{i}) = linspace(mingrid(i), maxgrid(i), opt.imgsz(i)+1);
    else
        opt.imgsz(i) = numel(opt.(fields{i})) - 1; % make imgsz consistent
    end
    opt.(pts_fields{i}) = opt.(fields{i})(1:end-1) + diff(opt.(fields{i}))/2;
end


xvec = opt.xvec;
yvec = opt.yvec;
zvec = opt.zvec;

