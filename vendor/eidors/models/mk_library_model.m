function out = mk_library_model(shape,elec_pos,elec_shape,maxsz,nfft,scale)
%MK_LIBRARY_MODEL - extruded FEM models based on curves in SHAPE_LIBRARY
%
% MK_LIBRARY_MODEL(shape,elec_pos,elec_shape,maxsz,nfft) where:
%   shape -  a cell array of strings and
%     shape{1} is the shape_library model used
%          (run shape_libary('list') to get a list
%     shape{2} is the the boundary. If absent, 'boundary' is assumed.
%     shape{3..} are strings specifying additional inclusions (such as
%     lungs)
%   shape may have a modifier
%      shape_name:ctr => center the shape at 0,0,0
%      shape_name:hctr => horizontal center shape
%      shape_name:vctr => vertical center shape
%   elec_pos - a vector specifying electrode positions. See
%     NG_MK_EXTRUDED_MODEL for details. To use the electrode positions
%     stored in the 'electrode' field in the shape_libary, specify elec_pos
%     as 'original'
%   elec_shape - a vector specifying electrode shapes. See
%     NG_MK_EXTRUDED_MODEL for details.
%   maxsz - maximum FEM size (default: course mesh)
%   nfft  - number of points to create along the boundary (default: 50)
%     If nfft==0, no interpolation takes place.
%   scale - avoids some Netgen issues by scaling the contours before
%     calling netgen and scaling the resulting model back afterwards
%     (default: 1). Note that electrode and maxh specifications are not
%     scaled.
%   if scale(2) is specified, this is the model height (default = 1)
%
% QUICK ACCESS TO COMMON MODELS:
%   MK_LIBRARY_MODEL(str) where str is a single string specifying a model.
%   Use MK_LIBRARY_MODEL('list') to obtain a list of available models.
%
% PATH TO LIBRARY MODELS
%   'LIBRARY_PATH' - get or set library path
%   mk_library_model LIBRARY_PATH => get library path
%   mk_library_model LIBRARY_PATH new_path => set library path
%   mk_library_model LIBRARY_PATH '' => don't store models
%
% See also: MK_EXTRUDED_MODEL, SHAPE_LIBRARY, MK_THORAX_MODEL, 
%           MK_HEAD_MODEL

% (C) 2011 Bartlomiej Grychtol. License: GPL version 2 or version 3
% $Id: mk_library_model.m 7057 2024-12-01 23:04:54Z aadler $

% Fill in defaults:
if nargin < 6; scale = 1;          end
if nargin < 5; nfft = 50;          end

if ischar(shape)
    switch shape
        case 'LIBRARY_PATH'
            switch nargin
                case 1
                    out = get_path;
                case 2
                    set_path(elec_pos);
            end
        case 'list'
            out = list_predef_model_strings;
        case 'UNIT_TEST'
            out = do_unit_test; return;
        otherwise
            fcol = find(shape==':');
            extra = false;
            if length(fcol)>=1;
               extra = shape(fcol+1:end);
               shape = shape(1:fcol-1);
            end
            out = predef_model(shape);
            switch extra
               case false; % pass
               case 'ctr'; 
                  out.nodes = out.nodes - mean(out.nodes);
               case 'hctr'; 
                  out.nodes(:,1:2) = out.nodes(:,1:2) - mean(out.nodes(:,1:2));
               case 'vctr'; 
                  out.nodes(:,3) = out.nodes(:,3) - mean(out.nodes(:,3));
               otherwise; error(['parameter "',extra,'" not recognized']);
            end
            out = mdl_normalize(out, 0); % not normalized by default
    end
else
   if ~iscell(shape)
      shape = {shape, 'boundary'};
   elseif numel(shape) == 1
      shape{2} = 'boundary';
   end
   fname = make_filename(shape,elec_pos,elec_shape,maxsz, nfft, scale);
   out = load_stored_model(fname);
   if ~isempty(out)
      return
   end
   s_shape = split_var_strings(shape(2:end));
   shapes = shape_library('get',shape{1},s_shape(1,:));
   if ~iscell(shapes), shapes = {shapes}; end
   %apply any indeces specified
   for i = 1:numel(shapes)
      eval(sprintf('shapes{i} = %f*shapes{i}%s;',scale(1),s_shape{2,i}));
   end
   if ischar(elec_pos) && strcmp(elec_pos,'original')
      el = shape_library('get',shape{1},'electrodes');
      electh= atan2(el(:,2),el(:,1))*180/pi;
      elec_pos = [electh,0.5*ones(size(electh))];
   end

   scaleH = prod(scale); % scale x height
   if nfft > 0
      [fmdl, mat_idx] = ng_mk_extruded_model({scaleH,shapes,[4,nfft],maxsz},...
         elec_pos,elec_shape);
   else
      [fmdl, mat_idx] = ng_mk_extruded_model({scaleH,shapes,0,maxsz},...
         elec_pos,elec_shape);
   end
   fmdl.nodes = fmdl.nodes/scale(1);
   fmdl.mat_idx = mat_idx;
   store_model(fmdl,fname)
   out = fmdl;
   out = mdl_normalize(out, 'default'); % not normalized by default
end




function out = load_stored_model(fname)
out = [];
if eidors_debug('query','mk_library_model')
   return
end
% If ng.opt has been specified, this this is a custom model:
% Don't load from store
if ng_write_opt('EXIST:ng.opt')
   eidors_msg('Custom ng.opt defined. Not using store',2);
   return
end
library_path = get_path();
if isempty(library_path); return; end

fname = [library_path '/' fname '.mat'];
if exist(fname,'file')
   eidors_msg('MK_LIBRARY_MODEL: Using stored model');
   load(fname);
   out = fmdl;
end

function store_model(fmdl,fname)
   if ng_write_opt('EXIST:ng.opt')
      eidors_msg('Custom ng.opt defined. Not saving to store',2);
      return
   end
   library_path = get_path();
   if isempty(library_path); return; end
   fname = [library_path '/' fname '.mat'];
   if exist('OCTAVE_VERSION');
      savver = '-v7';
   else
      savver = '-v7.3';
   end
   save(fname,savver,'fmdl');

function out = build_if_needed(cmd,str)
out = [];
out = load_stored_model(str);
if isempty(out)
   if ~iscell(cmd)
      cmd = {cmd};
   end
   for i = 1:length(cmd)
      if i ==1
         eval(['out = ' cmd{i} ';']);
      else
         eval(cmd{i});
      end
   end
   store_model(out,str);
end

%%%%%
% Lists predefined models (append when adding)
function out = list_predef_model_strings
out = {
    'adult_male_16el';
    'adult_male_32el';
    'adult_male_16el_lungs';
    'adult_male_32el_lungs';
    'adult_male_2x16el';
    'adult_male_2x32el';
    'adult_male_2x16el_lungs';
    'adult_male_2x32el_lungs';
% These are deprecated (but quietly supported). Use MK_THORAX_MODEL.
%     'adult_male_grychtol2016a_1x32';
%     'adult_male_grychtol2016a_2x16';
%     'adult_male_thorax_1x32';
%     'adult_male_thorax_1x16';
%     'adult_male_thorax_2x16';
%     'adult_female_thorax_1x32';
%     'adult_female_thorax_1x16';
%     'adult_female_thorax_2x16';
    'cylinder_16x1el_coarse';
    'cylinder_16x1el_fine';
    'cylinder_16x1el_vfine';
    'cylinder_16x2el_coarse';
    'cylinder_16x2el_fine';
    'cylinder_16x2el_vfine';
    'neonate_16el';
    'neonate_32el';
    'neonate_16el_lungs';
    'neonate_32el_lungs';
    'pig_23kg_16el';
    'pig_23kg_32el';
    'pig_23kg_16el_lungs';
    'pig_23kg_32el_lungs';
    'lamb_newborn_16el';
    'lamb_newborn_32el';
    'lamb_newborn_16el_lungs';
    'lamb_newborn_32el_lungs';
    'lamb_newborn_16el_organs';
    'lamb_newborn_32el_organs';
%     'lamb_newborn_32el_organs';
    'horse_16el';
    'horse_32el';
    'horse_2x16el';
    'horse_16el_lungs';
    'horse_32el_lungs';
    'horse_2x16el_lungs';
    'beagle_16el';
    'beagle_32el';
    'beagle_16el_lungs';
    'beagle_32el_lungs';
    'beagle_16el_rectelec';
    'beagle_32el_rectelec';
    'beagle_16el_lungs_rectelec';
    'beagle_32el_lungs_rectelec';
    };

%%%%%
% Use predefined model
function out = predef_model(str)
switch str
    case 'adult_male_16el'
        out = mk_library_model({'adult_male','boundary'},...
            [16 1 0.5],[0.05],0.08);
    case 'adult_male_2x16el'
        elayers = [0.65,0.35];
        out = mk_library_model({'adult_male','boundary'},...
            [16 1 elayers],[0.05],0.08);
    case 'adult_male_32el'
        out = mk_library_model({'adult_male','boundary'},...
            [32 1 0.5],[0.05],0.08);
    case 'adult_male_2x32el'
        elayers = [0.65,0.35];
        out = mk_library_model({'adult_male','boundary'},...
            [32 1 elayers],[0.05],0.08);
    case 'adult_male_16el_lungs'
        out = mk_library_model({'adult_male','boundary','left_lung','right_lung'},...
            [16 1 0.5],[0.05],0.08);
    case 'adult_male_2x16el_lungs'
        elayers = [0.65,0.35];
        out = mk_library_model({'adult_male','boundary','left_lung','right_lung'},...
            [16 1 elayers],[0.05],0.08);
    case 'adult_male_32el_lungs'
        out = mk_library_model({'adult_male','boundary','left_lung','right_lung'},...
            [32 1 0.5],[0.05],0.08);
    case 'adult_male_2x32el_lungs'
        elayers = [0.65,0.35];
        out = mk_library_model({'adult_male','boundary','left_lung','right_lung'},...
            [32 1 elayers],[0.05],0.08);

    case {'adult_male_grychtol2016a_1x32'
          'adult_male_grychtol2016a_2x16'
          'adult_male_thorax_1x32]'
          'adult_male_thorax_1x16'
          'adult_male_thorax_2x16'
          'adult_female_thorax_1x32'
          'adult_female_thorax_1x16' 
          'adult_female_thorax_2x16'}
       warning('EIDORS:DeprecatedInterface', ...
          ['MK_LIBRARY_MODEL(''%s'') is deprecated. '...
           'Use mk_thorax_model(''%s'') instead.'], str, str);
       out = mk_thorax_model(str);
       try out = out.fwd_model; end

    case 'cylinder_16x1el_coarse'
       out = build_if_needed(...
          'ng_mk_cyl_models([10,15],[16,5],[0.5,0,0.18])', str);
    case 'cylinder_16x1el_fine'
       out = build_if_needed(...
          'ng_mk_cyl_models([10,15,1.1],[16,5],[0.5,0,0.15])',str);
    case 'cylinder_16x1el_vfine'
        out = build_if_needed(...
           'ng_mk_cyl_models([10,15,0.8],[16,5],[0.5,0,0.08])',str);
    case 'cylinder_16x2el_coarse'
        out = build_if_needed(...
           'ng_mk_cyl_models([30,15],[16,10,20],[0.5,0,0.18])',str);
    case 'cylinder_16x2el_fine'
        out = build_if_needed(...
           'ng_mk_cyl_models([30,15,1.5],[16,10,20],[0.5,0,0.15])',str);
    case 'cylinder_16x2el_vfine'
        out = build_if_needed(...
           'ng_mk_cyl_models([30,15,0.8],[16,10,20],[0.5,0,0.08])',str);


    case 'neonate_16el'
        out = mk_library_model({'neonate','boundary'},[16 1 0.5],[0.1 0 -1 0 60],0.08,49);
    case 'neonate_32el'
        out = mk_library_model({'neonate','boundary'},[32 1 0.5],[0.06 0 -1 0 60],0.08,49);
    case 'neonate_16el_lungs'
        out = mk_library_model({'neonate','boundary','left_lung','right_lung'},[16 1 0.5],[0.1 0 -1 0 60],0.08,49);
    case 'neonate_32el_lungs'
        out = mk_library_model({'neonate','boundary','left_lung','right_lung'},[32 1 0.5],[0.06 0 -1 0 60],0.08,49);

    case 'pig_23kg_16el'
        out = mk_library_model({'pig_23kg','boundary'},...
            [16 1 0.5],[0.05 0 -1 0 60],0.08);
    case 'pig_23kg_32el'
        out = mk_library_model({'pig_23kg','boundary'},...
            [32 1 0.5],[0.05 0 -1 0 60],0.08);
    case 'pig_23kg_16el_lungs'
        out = mk_library_model({'pig_23kg','boundary','lungs(1:2:end,:)'},...
            [16 1 0.5],[0.05 0 -1 0 60],0.08);
    case 'pig_23kg_32el_lungs'
        out = mk_library_model({'pig_23kg','boundary','lungs(1:2:end,:)'},...
            [32 1 0.5],[0.05 0 -1 0 60],0.08);

    case 'lamb_newborn_16el'
%        out = build_if_needed(...
%           {['ng_mk_extruded_model({208,208*',...
%             'shape_library(''get'',''lamb_newborn'',''boundary'')',...
%             ',0,10},[16,1.995,104],[1])'],'out.nodes = out.nodes/204;'}, ...
%           str);
         out = mk_library_model({'lamb_newborn','boundary'},[16,1.995,104],[1],10,0,208);
    case 'lamb_newborn_32el'
         % Very sensitive to the .980 offset. This is the only I can find that works.
         out = mk_library_model({'lamb_newborn','boundary'},[32,1.980,104],[1],15,50,208);
         out.electrode = out.electrode([2:32,1]);
    case 'lamb_newborn_16el_organs'
       out = mk_library_model({'lamb_newborn','boundary','lungs','heart'},[16,1.995,104],[1],10,0,208);
    case 'lamb_newborn_16el_lungs'
       out = mk_library_model({'lamb_newborn','boundary','lungs'},[16,1.995,104],[1],10,0,208);
    case 'lamb_newborn_32el_lungs'
         % Very sensitive to the .980 offset. This is the only I can find that works.
         out = mk_library_model({'lamb_newborn','boundary','lungs'},[32,1.980,104],[1],15,50,208);
         out.electrode = out.electrode([2:32,1]);
    case 'lamb_newborn_32el_organs'
         % Very sensitive to the .980 offset. This is the only I can find that works.
         out = mk_library_model({'lamb_newborn','boundary','lungs','heart'},[32,1.980,104],[1],15,50,208);
         out.electrode = out.electrode([2:32,1]);
%     case 'lamb_newborn_32el_organs'
    case 'horse_16el';
% Horse models are on their back. Ventral is up
      scale = [100,.35 ]; Nelec=16;
      elhig = prod(scale)/2;
      out = mk_library_model({'horse','boundary'},[Nelec,1.51,elhig],[2,6,0.5],2,50,scale);
    case 'horse_32el';
      scale = [100,.35 ]; Nelec=32;
      elhig = prod(scale)/2;
      out = mk_library_model({'horse','boundary'},[Nelec,1.51,elhig],[2,6,0.5],2,50,scale);
    case 'horse_2x16el';
      scale = [100,.35 ]; Nelec=16;
      elhig = [5,3]*prod(scale)/8;
      out = mk_library_model({'horse','boundary'},[Nelec,1.51,elhig],[2,3,0.5],2,50,scale);
    case 'horse_16el_lungs';
      scale = [100,.5 ]; Nelec=16;
      elhig = prod(scale)/2;
      out = mk_library_model({'horse','boundary','left_lung','right_lung','heart'},[Nelec,1.51,elhig],[2,6,0.5],2,50,scale);
    case 'horse_32el_lungs';
      scale = [100,.35 ]; Nelec=32;
      elhig = prod(scale)/2;
      out = mk_library_model({'horse','boundary','left_lung','right_lung','heart'},[Nelec,1.51,elhig],[2,6,0.5],2,50,scale);
    case 'horse_2x16el_lungs';
      scale = [100,.5 ]; Nelec=16;
      elhig = [5,3]*prod(scale)/8;
      out = mk_library_model({'horse','boundary','left_lung','right_lung','heart'},[Nelec,1.51,elhig],[2,3,0.5],2,50,scale);
    case 'beagle_16el';
      scale = 49;
      out = mk_library_model({'beagle','boundary'}, ...
         [16 1 scale*0.5],[2,0,0.10],10,0,49);
    case 'beagle_32el';
      scale = 49;
      out = mk_library_model({'beagle','boundary'}, ...
         [32 1 scale*0.5],[2,0,0.10],10,0,49);
    case 'beagle_16el_rectelec';
      scale = 49;
      out = mk_library_model({'beagle','boundary'}, ...
         [16 1 scale*0.5],8*[0.25,1,0.05],10,0,49);
    case 'beagle_32el_rectelec';
      scale = 49;
      out = mk_library_model({'beagle','boundary'}, ...
         [16 1 scale*0.5],8*[0.25,1,0.05],10,0,49);

    case 'beagle_16el_lungs';
      scale = 49;
      out = mk_library_model({'beagle','boundary','left_lung','right_lung'}, ...
         [16 1 scale*0.5],[2,0,0.10],10,0,49);

    case 'beagle_16el_lungs_rectelec';
      scale = 49;
      out = mk_library_model({'beagle','boundary','left_lung','right_lung'}, ...
         [16 1 scale*0.5],8*[0.25,1,0.05],10,0,49);

    case 'beagle_32el_lungs';
      scale = 49;
      out = mk_library_model({'beagle','boundary','left_lung','right_lung'}, ...
         [32 1 scale*0.5],[1.5,0,0.10],10,0,49);

    case 'beagle_32el_lungs_rectelec';
      scale = 49;
      out = mk_library_model({'beagle','boundary','left_lung','right_lung'}, ...
         [32 1 scale*0.5],8*[0.25,1,0.05],10,0,49);

    otherwise
        error('No such model');
end
%give the model a name
out.name = str;


function str = make_filename(shape, elec_pos, elec_shape, ...
                             maxsz, nfft, scale);
%at this point, shape is a cell array of strings e.g. {'pig_23kg','lungs')
str = shape{1};
shape(1) = []; %remove the first element
shape = sort(shape); %sort the others
for i = 1:numel(shape)
    str = [str '_' shape{i}];
end
str = [str '_EP'];
for i = 1:numel(elec_pos)
    str = [str '_' num2str(elec_pos(i))];
end
str = [str '_ES'];
if ischar(elec_shape)
    str = [str '_' elec_shape];
else
    for i = 1:numel(elec_shape)
        str = [str '_' num2str(elec_shape(i))];
    end
end
if ~isempty(maxsz)
    str = [str '_maxsz_' num2str(maxsz)];
end
if ~isempty(nfft)
    str = [str '_nfft_' num2str(nfft)];
end
if ~isempty(scale)
    str = [str '_scale' sprintf('%f_',scale)];
end

%remove colons
str = strrep(str,':','-');

function clean = split_var_strings(strc)
for i = 1:numel(strc)
    [clean{1,i} clean{2,i}] = strtok(strc{i},'([{');
end


function out = get_path
global eidors_objects
out = eidors_objects.model_cache;

function set_path(val)
global eidors_objects
eidors_objects.model_cache = val;
if isempty(val); return; end % val=='' means don't use
%if folder doesn't exist, create it
if ~exist(val,'dir')
    ver= eidors_obj('interpreter_version');
    if ver.ver<4 || ver.ver>=7
       mkdir(val);
    else
 % matlab 6.x has a stupid mkdir function
       system(['mkdir ',val]);
    end
end

function out = do_unit_test
models = mk_library_model('list');
deprecated = {'adult_male_grychtol2016a_1x32'
          'adult_male_grychtol2016a_2x16'
          'adult_male_thorax_1x32]'
          'adult_male_thorax_1x16'
          'adult_male_thorax_2x16'
          'adult_female_thorax_1x32'
          'adult_female_thorax_1x16' 
          'adult_female_thorax_2x16'};
if exist('OCTAVE_VERSION');
  deprecated={}; % Octave doesn't have triangulation
end
models = vertcat(deprecated, models);
n_models = numel(models);
sqrt_n_models = ceil(sqrt(n_models));
error_list = {};
for i = 1:numel(models)
    eidors_msg('\n\n\n DOING  MODEL (%s)\n\n\n',models{i},0);
    try
        mdl = mk_library_model(models{i});
    catch
        error_list = [error_list, models(i)];
        continue;
    end
    img = mk_image(mdl,1);
    try
        n = numel(mdl.mat_idx);
    catch
        n =1;
    end
    if n >1
        for j = 2:n
            img.elem_data(mdl.mat_idx{j}) = 0.25;
        end
    end
    subplot(sqrt_n_models, sqrt_n_models,i);
    show_fem(img,[0,1,0]); axis off;
    title(models{i},'Interpreter','none');
    drawnow
end

if ~isempty(error_list)
    disp('There were errors in generating the following models:')
    disp(error_list);
end

out = mk_library_model('pig_23kg_16el');
unit_test_cmp('pig_23kg_16el', max(out.nodes),[1.004200000000000,0.998200000000000,1.000000000000000],1e-10);

out = mk_library_model('pig_23kg_16el:ctr');
unit_test_cmp('pig_23kg_16el:ctr', max(out.nodes),[0.995060653391834,1.064458637676478,0.501929982309894],1e-10);

out = mk_library_model('pig_23kg_16el:hctr');
unit_test_cmp('pig_23kg_16el:hctr', max(out.nodes),[0.995060653391834,1.064458637676478,1.000000000000000],1e-10);

out = mk_library_model('pig_23kg_16el:vctr');
unit_test_cmp('pig_23kg_16el:vctr', max(out.nodes),[1.004200000000000,0.998200000000000,0.501929982309894],1e-10);

out = mk_library_model({'pig_23kg','boundary','lungs(1:2:end,:)'},[32 1 0.5],[0.05 0 -1 0 60],0.08);
unit_test_cmp('pig_23kg', max(out.nodes),[1.004200000000000 0.998200000000000 1.000000000000000],1e-10);

