function out = mdl_normalize(mdl, val)
%MDL_NORMALIZE Check or set the normalize_measurements flag on a model.
%  OUT = MDL_NORMALIZE(MDL) returns the value of the 
%  'normalize_measurements' or 'normalize' field on MDL. If absent, the 
%  default value is returned as set with EIDORS_DEFAULT. 
%
%  MDL = MDL_NORMALIZE(MDL, VAL) sets the 'normalize_measurements' to VAL
%
%  MDL = MDL_NORMALIZE(MDL, 'default') uses the EIDORS_DEFAULT value to set
%  set the 'normalize_measurements' flag on the model.
%
%  OUT = MDL_NORMALIZE('default') returns the EIDORS_DEFAULT value to set
%  set the 'normalize_measurements' flag on the model.
%
% See also: FWD_MODEL_PARAMETERS, EIDORS_DEFAULT, EIDORS_STARTUP

% (C) 2012 Bartlomiej Grychtol. License: GPL v2 or v3
% $Id: mdl_normalize.m 7043 2024-11-30 14:46:20Z aadler $

if ischar(mdl) && strcmp(mdl, 'UNIT_TEST'); do_unit_test; return; end
if ischar(mdl) && strcmp(mdl, 'default') 
   out = eidors_default; 
   return
end

% check that we got a fwd_model
try 
   switch mdl.type
      case 'fwd_model'
         % do nothing
      case 'image'
         img = mdl;
         mdl = img.fwd_model;
      otherwise
         error('EIDORS:WrongInput','Expected fwd_model or image.');
   end
catch
   warning('EIDORS:UnknownType','Object has no type. Assuming fwd_model.');
end
% do the real thing
switch nargin
    case 1
        out = get_flag(mdl);
        if isempty(out)
            out = eidors_default;
            warning('EIDORS:normalize_flag_not_set', ...
                'normalize_measurements flag not set on model. Using default: %d',out);
        end
    case 2
        if ischar(val) && strcmp(val, 'default')
           val = eidors_default;
        end
        out = set_flag(mdl,val);
        if exist('img','var') % input was an image
           img.fwd_model = out;
           out = img;
        end
    otherwise
        error('Wrong number of parameters');
end



function out = get_flag(mdl)
 out = [];
% iterate over fields
ff = fieldnames(mdl);
for i = 1:length(ff);
    % if both are set, you're in trouble
    if any(strcmp(ff{i},{'normalize','normalize_measurements'}))
        out = mdl.(ff{i});
    else
        % won't hit 'normal' or anything that starts with 'normals'. 
        token = regexp(ff{i},'normal[^s]+.*','match','once'); 
        if ~isempty(token);
            warning('Suspect field "%s" found on the model',token);
        end
        
    end
end

function mdl = set_flag(mdl, val)
mdl = eidors_obj('set',mdl, 'normalize_measurements', val);

function do_unit_test
s = warning('query', 'backtrace');
warning off backtrace;
eidors_default('set','mdl_normalize',@(x) 0);
mdl = mk_common_model('c2t2',16);
fmdl = mdl.fwd_model;
unit_test_cmp('Does it come normalized?', mdl_normalize(fmdl), 0);
fmdl = mdl_normalize(fmdl,1);
unit_test_cmp('Normalize it. How about now?', mdl_normalize(fmdl), 1);
fmdl = mdl_normalize(fmdl,0);
unit_test_cmp('De-normalize it. And now?', mdl_normalize(fmdl), 0);
fmdl = rmfield(fmdl,'normalize_measurements');
eidors_msg(mdl_normalize(fmdl));
unit_test_cmp('Remove the flag', ...
   lastwarn, 'normalize_measurements flag not set on model. Using default: 0');
fmdl.normalise_measurements = 1; % s, throw a warning, return default
eidors_msg(mdl_normalize(fmdl));
unit_test_cmp('Spelling error', ...
   lastwarn, 'normalize_measurements flag not set on model. Using default: 0');
fmdl.normalize_my_gran = 1; % throw a warning, return default
eidors_msg(mdl_normalize(fmdl));
unit_test_cmp('And now let''s set it to something stupid', ...
   lastwarn, 'normalize_measurements flag not set on model. Using default: 0');
fmdl.normalize = 1; % accept happily
unit_test_cmp('Accept happily', mdl_normalize(fmdl), 1);
eidors_msg('But an imdl is not acceptable!');
try
    eidors_msg(mdl_normalize(mdl));
catch
    unit_test_cmp('Just so.', 1,1);
end
warning(s);
