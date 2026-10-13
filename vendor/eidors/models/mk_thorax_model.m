function out = mk_thorax_model(str, varargin)
%MK_THORAX_MODEL FEM models of the thorax
%
% MK_THORAX_MODEL provides a shorthand to predefined thorax FEMs and 
% for using contributed models with PLACE_ELEC_ON_SURF. You may be asked
% to download missing files. Specialized MK_THORAX_MODEL_* functions
% provide additional presets and options.
%
% MK_THORAX_MODEL(shapestr,elec_pos, elec_shape, maxh) where:
%  shapestr    - a string specifying the underlying model.
%                Run MK_THORAX_MODEL('list_shapes') for a list.
%  elec_pos    - a vector specifying electrode positions.
%                See PLACE_ELEC_ON_SURF for details.
%  elec_shape  - a vector specifying electrode shapes.
%                See PLACE_ELEC_ON_SURF for details.
%  This usage returns a fwd_model structure.
%
% MK_THORAX_MODEL(modelstr) provides quick access to predefined models.
% Run MK_THORAX_MODEL('list') for a list. This usage returns either a 
% fwd_model or an image structure. Specialized MK_THORAX_MODEL_* functions
% provide additional presets.
%
% MK_THORAX_MODEL('list_shapes') lists available thorax shapes without 
%  electrodes.
%
% MK_THORAX_MODEL('list') lists available predefined models.
%
%
% See also: MK_THORAX_MODEL_BP3D, MK_THORAX_MODEL_GRYCHTOL2016a, 
% PLACE_ELEC_ON_SURF, MK_LIBRARY_MODEL

% (C) 2015-2024 Bartlomiej Grychtol. License: GPL version 2 or 3
% $Id: mk_thorax_model.m 7061 2024-12-08 16:40:33Z aadler $

if nargin>0 && ischar(str) 
   switch(str)
      case 'UNIT_TEST';
         do_unit_test; return;
      case 'list_shapes'
         out = list_basic_shapes; return;
      case 'list'
         out = list_predef_models; return;
   end
end

opt.fstr = 'mk_thorax_model';
out = eidors_cache(@do_mk_thorax_model,[{str}, varargin(:)'],opt);
switch out.type
   case 'image'
      out.fwd_model = mdl_normalize(out.fwd_model, 'default');
   case 'fwd_model'
      out = mdl_normalize(out, 'default');
   otherwise
      error('Produced struct with unrecognized type');
end
end

function out = do_mk_thorax_model(str, elec_pos, elec_shape, maxh)

if ismember(str,list_predef_models)
   out = build_predef_model(str);
   return
end

if ~ismember(str, list_basic_shapes)
   error('Shape str "%s" not understood.',str);
end

if ~any(str=='-') % basic shaps have no hyphens
   out = build_basic_model(str);
   if nargin > 1
      out = place_elec_on_surf(out,elec_pos,elec_shape,[],maxh);
      out = fix_boundary(out);
   end
else
   out = build_advanced_model(str, elec_pos, elec_shape, maxh);
end
end

function ls = list_basic_shapes
   ls = {'male','female','bp3d','bp3d-organs'};
end

function out = build_basic_model(str)
switch(str)
   case {'male', 'female'}
      out = eidors_cache(@remesh_at_model, {str});
   case 'bp3d'
      out = mk_thorax_model_bp3d();
end
end

function out = build_advanced_model(str, elec_pos, elec_shape, maxh)
switch(str)
   case {'bp3d-organs'}
      out = mk_thorax_model_bp3d(elec_pos, elec_shape, maxh, 'fine');
end
end

function out = remesh_at_model(str)
   tmpnm  = tempname;
   
   stlfile = [tmpnm '.stl'];
   volfile = [tmpnm '.vol'];
   
   contrib = 'at-thorax-mesh'; file =  sprintf('%s_t_mdl.mat',str);
   load(get_contrib_data(contrib,file));
   if strcmp(str,'male')
      fmdl.nodes = fmdl.nodes - 10; % center the model
   end
   fmdl = fix_boundary(fmdl);
   STL.vertices = fmdl.nodes;
   STL.faces    = fmdl.boundary;
   stl_write(STL,stlfile, 'txt'); % netgen only works well with txt files
   opt.stloptions.yangle = 55;
   opt.stloptions.edgecornerangle = 5;
   opt.meshoptions.fineness = 6;
   opt.options.meshsize = 30;
   opt.options.minmeshsize = 10;
   opt.stloptions.resthsurfcurvfac = 2;
   opt.stloptions.resthsurfcurvenable = 1;
   opt.stloptions.chartangle = 30;
   opt.stloptions.outerchartangle = 90;
   ng_write_opt(opt);
   call_netgen(stlfile,volfile);
   out=ng_mk_fwd_model(volfile,[],[],[],[]);
   delete(stlfile);
   delete(volfile);
   delete('ng.opt');
   
   out = rmfield(out,...
      {'mat_idx','boundary_numbers'});
   
end


function ls = list_predef_models
   ls = {'adult_male_bp3d_1x32'
         'adult_male_bp3d_2x16'
         'adult_male_bp3d_1x32_organs'
         'adult_male_bp3d_2x16_organs'
         'adult_male_grychtol2016a_1x32'
         'adult_male_grychtol2016a_2x16'
         'adult_male_thorax_1x32'
         'adult_male_thorax_1x16'
         'adult_male_thorax_2x16'
         'adult_female_thorax_1x32'
         'adult_female_thorax_1x16'
         'adult_female_thorax_2x16'};
end

function out = build_predef_model(str)
   switch str
      case 'adult_male_grychtol2016a_1x32'
         out = mk_thorax_model_grychtol2016a('1x32_ring');
         
      case 'adult_male_grychtol2016a_2x16'
         out = mk_thorax_model_grychtol2016a('2x16_planar');

      case 'adult_male_bp3d_1x32'
         out = mk_thorax_model_bp3d('1x32_ring');

      case 'adult_male_bp3d_2x16'
         out = mk_thorax_model_bp3d('2x16_planar');

      case 'adult_male_bp3d_1x32_organs'
         out = mk_thorax_model_bp3d('1x32_ring', 'fine');

      case 'adult_male_bp3d_2x16_organs'
         out = mk_thorax_model_bp3d('2x16_planar', 'fine');

      case 'adult_male_thorax_1x32'
         eth32 = cumsum([0.1 0.2*ones(1,15) 0.25 0.2*ones(1,15)])/6.45;
         ep = 360*eth32' - 90; ep(:,2) = 175;
         out = mk_thorax_model('male',ep,[5 0 .5],10);
      case 'adult_male_thorax_1x16'
         eth16 = cumsum([0.2 0.4*ones(1,7) 0.5 0.4*ones(1,7)])/6.5;
         ep = 360*eth16' - 90; ep(:,2) = 175;
         out = mk_thorax_model('male',ep,[5 0 .5],10);
      case 'adult_male_thorax_2x16'
         eth16 = cumsum([0.2 0.4*ones(1,7) 0.5 0.4*ones(1,7)])/6.5;
         ep1= 360*eth16' - 90; ep1(:,2) = 200;
         ep2= 360*eth16' - 90; ep2(:,2) = 150;
         out = mk_thorax_model('male',[ep1;ep2],[5 0 .5],10);
      case 'adult_female_thorax_1x32'
         eth32 = cumsum([0.1 0.2*ones(1,15) 0.25 0.2*ones(1,15)])/6.45;
         ep = 360*eth32' - 90; ep(:,2) = 175;
         out = mk_thorax_model('female',ep,[5 0 .5],10);
      case 'adult_female_thorax_1x16'
         eth16 = cumsum([0.2 0.4*ones(1,7) 0.5 0.4*ones(1,7)])/6.5;
         ep = 360*eth16' - 90; ep(:,2) = 175;
         out = mk_thorax_model('female',ep,[5 0 .5],10);
      case 'adult_female_thorax_2x16'
         eth16 = cumsum([0.2 0.4*ones(1,7) 0.5 0.4*ones(1,7)])/6.5;
         ep1= 360*eth16' - 90; ep1(:,2) = 250;
         ep2= 360*eth16' - 90; ep2(:,2) = 175;
         out = mk_thorax_model('female',[ep1;ep2],[5 0 .5],10);
   end
end


function do_unit_test
   clf
   test_predef_models;
   figure
   subplot(221)
   show_fem(mk_thorax_model('male'));
   
   subplot(222)
   show_fem(mk_thorax_model('female'));

   subplot(223)
   eth16 = 360*cumsum([0.2 0.4*ones(1,7) 0.5 0.4*ones(1,7)])/6.5 - 90; eth16 = eth16';
   eth32 = 360*cumsum([0.1 0.2*ones(1,15) 0.25 0.2*ones(1,15)])/6.45 - 90; eth32 = eth32';
   ep = eth16; ep(:,2) = 150;
   ep(17:48,1) = eth32; ep(17:48,2) = 175;
   ep(49:64,1) = eth16; ep(49:64,2) = 200;
   mdl = mk_thorax_model('male',ep,[5 0 .5],10);
   show_fem(mdl);

   subplot(224)
   mdl = mk_thorax_model('female',ep,[5 0 .5],10);
   show_fem(mdl);

   % This doens't work with mk_thorax_model
%  ng_write_opt('MSZCYLINDER',[0,0,50,0,0,60,180,5]);
%  show_fem(mk_thorax_model('male'));
   
end

function test_predef_models
   models = mk_thorax_model('list');
   n_models = numel(models);
%    sqrt_n_models = ceil(sqrt(n_models));
   error_list = {};
   for i = 1:numel(models)
      eidors_msg('\n\n\n DOING  MODEL (%s)\n\n\n',models{i},0);
      try
         mdl = mk_thorax_model(models{i});
      catch
         error_list = [error_list, models(i)];
         continue;
      end
   
%       subplot(sqrt_n_models, sqrt_n_models,i);
      show_fem(mdl,[0,1,0]); axis off;
      title(models{i},'Interpreter','none');
      drawnow
   end
   if ~isempty(error_list)
      disp('There were errors in generating the following models:')
      disp(error_list);
   end
end
