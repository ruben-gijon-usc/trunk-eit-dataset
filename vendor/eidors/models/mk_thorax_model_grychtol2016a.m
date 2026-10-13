function img = mk_thorax_model_grychtol2016a(stimpat)
% Builds the male thorax models from Grychtol, Müller and Adler (2016)
% Accepts a string parameter:
%    '2x16_planar'   - 2 rings of 16 electrodes, planar stimulation
%    '2x16_odd-even' - 2 rings, each stimulation is cross-plane 
%    '2x16_square'   - 2 rings, every second stimulation is cross-plane
%    '2x16_adjacent' - like square, but with adjacent stimulation
%    '1x32_ring'     - single ring of 32 electrodes
%
% CITATION_REQUEST:
% AUTHOR: Bartlomiej Grychtol, Beat Müller and Andy Adler
% TITLE: 3D EIT image reconstruction with GREIT
% JOURNAL: Physiological Measurement
% VOL: 37
% YEAR: 2016

% (C) 2016-2024 Bartlomiej Grychtol. License: GPL version 2 or 3
% $Id: mk_thorax_model_grychtol2016a.m 6908 2024-05-29 09:04:56Z bgrychtol $

citeme(mfilename);

if nargin<1 || isempty(stimpat)
   stimpat = '1x32_ring';
end

fname = [get_cache_path filesep 'adult_male_grychtol2016a.mat'];
if exist(fname,'file')
   eidors_msg('@@ Using stored model.', 3);
   load(fname, 'img');
else
   img = build_image;
   save(fname, 'img');
end
img.name = img.fwd_model.name;
img.fwd_model.name = ...
   sprintf(['Thorax model from Grychtol, Müller and Adler (2016) '...
   '- %s electrode config'],stimpat);
img = mdl_normalize(img, 'default');
img = set_predef_stim(img, stimpat);

end

function img = build_image
   eth16 = 360*cumsum([0.2 0.4*ones(1,7) 0.5 0.4*ones(1,7)])/6.5 - 90; eth16 = eth16';
   eth32 = 360*cumsum([0.1 0.2*ones(1,15) 0.25 0.2*ones(1,15)])/6.45 - 90; eth32 = eth32';
   ep = eth16; ep(:,2) = 150;
   ep(17:48,1) = eth32; ep(17:48,2) = 175;
   ep(49:64,1) = eth16; ep(49:64,2) = 200;
   mdl = mk_thorax_model('male',ep,[5 0 .5],10);
   mdl.name = sprintf(['Thorax mesh from Grychtol, Müller and Adler (2016)']);
   
   
   opt = organ_options;
   img = mk_lung_image(mdl,opt);
   img.fwd_model = mdl_normalize(img.fwd_model, 'default');
   
end

function  opt = organ_options
   opt.bkgnd_val = 1  ;
   opt.lung_val =  .2;
   opt.heart_val = 1.5;
   opt.left_lung_center =  [ 75 25 100];
   opt.right_lung_center = [-75 25 100];
   opt.left_lung_axes = [80 100 250];
   opt.right_lung_axes = [80 100 250];
   opt.heart_center = [20 -25 200];
   opt.heart_axes = [50 60 75];
   opt.diaphragm_center = [0 -50 0];
   opt.diaphragm_axes = [220 190 120];
end

function out = get_cache_path
   global eidors_objects
   out = eidors_objects.model_cache;
end