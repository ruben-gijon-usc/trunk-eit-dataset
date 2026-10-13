function out = mk_head_model(str, varargin)
%MK_HEAD_MODEL FEM models of the head
%
% MK_HEAD_MODEL provides a shorthand to predefined head FEMs and 
% for using contributed models with PLACE_ELEC_ON_SURF. You may be asked
% to download missing files. Specialized MK_HEAD_MODEL_* functions provide
% additional presets and options.
%
% MK_HEAD_MODEL(shapestr,elec_pos, elec_shape, maxh) where:
%  shapestr    - a string specifying the underlying model.
%                Run MK_HEAD_MODEL('list_shapes') for a list.
%  elec_pos    - a vector specifying electrode positions.
%                See PLACE_ELEC_ON_SURF for details.
%  elec_shape  - a vector specifying electrode shapes.
%                See PLACE_ELEC_ON_SURF for details.
%  This usage returns a fwd_model structure.
%
% MK_HEAD_MODEL(modelstr) provides quick access to predefined models.
% Run MK_HEAD_MODEL('list') for a list. This usage returns either a 
% fwd_model or an image structure. Specialized MK_HEAD_MODEL_* functions
% provide additional presets.
%
% MK_HEAD_MODEL('list_shapes') lists available thorax shapes without 
%  electrodes.
%
% MK_HEAD_MODEL('list') lists available predefined models.
%
% See also: MK_HEAD_MODEL_ADULT, PLACE_ELEC_ON_SURF

% (C) 2015-2024 Bartlomiej Grychtol. License: GPL version 2 or 3
% $Id: mk_head_model.m 7124 2024-12-29 15:18:32Z aadler $

if nargin>0 && ischar(str) 
   switch(str)
      case 'UNIT_TEST'
         do_unit_test; return;
      case 'list_shapes'
         out = list_basic_shapes; return;
      case 'list'
         out = list_predef_models; return;
   end
end

opt.fstr = 'mk_head_model';
out = eidors_cache(@do_mk_head_model,[{str}, varargin(:)'],opt);
switch out.type
   case 'image'
      out.fwd_model = mdl_normalize(out.fwd_model, 'default');
   case 'fwd_model'
      out = mdl_normalize(out, 'default');
   otherwise
      error('Produced struct with unrecognized type');
end
end

function out = do_mk_head_model(str, elec_pos, elec_shape, maxh)

if ismember(str,list_predef_models)
   out = build_predef_models(str);
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
   ls = {'adult', 'adult-organs'};
end

function out = build_basic_model(str)
switch(str)
   case 'adult'
      out = mk_head_model_adult();
end
end

function out = build_advanced_model(str, elec_pos, elec_shape, maxh)
switch(str)
   case {'adult-organs'}
      out = mk_head_model_adult(elec_pos, elec_shape, maxh, 'coarse');
end
end

function ls = list_predef_models
   ls = {'adult_10-10'
         'adult_2x16'
         'adult_1x32'
        };
end

function out = build_predef_models(str)
switch str
   case 'adult_10-10'
      out = mk_head_model_adult('10-10','coarse');
   case 'adult_2x16'
      out = mk_head_model_adult('2x16_planar','coarse');
   case 'adult_1x32'
      out = mk_head_model_adult('1x32_ring', 'coarse');
end
end

function do_unit_test
test_predef_models
end

function test_predef_models
   models = mk_head_model('list');
   n_models = numel(models);
%    sqrt_n_models = ceil(sqrt(n_models));
   error_list = {};
   for i = 1:numel(models)
      eidors_msg('\n\n\n DOING  MODEL (%s)\n\n\n',models{i},0);
      try
         mdl = mk_head_model(models{i});
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

