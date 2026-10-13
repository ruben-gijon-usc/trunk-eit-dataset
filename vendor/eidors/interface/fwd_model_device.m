function fmdl = fmdl_model_device(fmdl,device, opt);
% FMDL_MODEL_DEVICE: set vendor/device specific parameters
% fmdl_model_device is a utility function to set vendor and device specific parameters.
% It sets the 'stimulation' and 'normalize_measurements' fields to the vendor-typical values
%
% Usage:
% fmdl = fmdl_model_device(fmdl,device, opt);
% 
% Devices and Options:
%  Draeger/Pulmovista:
%    - sets adjacent, rotate_meas, and mdl_normalize=true
%  Sentec/Swisstom
%    - Default: opt.skip = 4; opt.next = 1;
%    - sets skip pattern, meas_current_next, and mdl_normalize=false

% (C) 2024 Andy Adler License: GPL ver 2 or 3
% $Id: fwd_model_device.m 7002 2024-11-24 13:11:35Z aadler $

if ischar(fmdl) && strcmp(fmdl,'UNIT_TEST'); do_unit_test(); return; end

if nargin<3; opt=struct(); end

switch lower(device)
  case {'draeger','dräger','pulmovista500'}
    fmdl=mdl_normalize(fmdl,true);
    mA = 0.01; % Estimate from docs
    fmdl.stimulation = mk_stim_patterns(16,1, ...
       [0,1],[0,1],{'no_meas_current','rotate_meas'},mA);

  case {'sentec','swisstom'}

    if ~isfield(opt,'skip'); opt.skip = 4; end
    if ~isfield(opt,'next'); opt.next = 1; end
    fmdl=mdl_normalize(fmdl,false);
    mA = 0.01; % Estimate from docs
    SS = [0,opt.skip+1];
    fmdl.stimulation = mk_stim_patterns(32,1, ...
       SS,SS,{sprintf('no_meas_current_next%d',opt.next)},mA);

  otherwise error('No information about device');
end

function do_unit_test
   fmdl = getfield(mk_common_model('a2c2',16), ...
                   'fwd_model');
   ssmm = stim_meas_list(fmdl.stimulation);
   unit_test_cmp('Init',ssmm([8,35],:), [
       1  2 11 10; 3  4 13 12]);

   fmdl = fwd_model_device(fmdl,'Draeger');
   ssmm = stim_meas_list(fmdl.stimulation);
   unit_test_cmp('Draeger',size(ssmm,1),208);
   unit_test_cmp('Draeger',ssmm([8,35],:), [
       1  2 11 10; 3  4 14 13]);
   unit_test_cmp('Draeger',fmdl.normalize_measurements, true);

 
   fmdl = getfield(mk_common_model('b2c2',32), ...
                   'fwd_model');
   fmdl = fwd_model_device(fmdl,'Sentec');
   ssmm = stim_meas_list(fmdl.stimulation);
   unit_test_cmp('Sentec',size(ssmm,1),736);
   unit_test_cmp('Sentec',ssmm([8,35],:), [
       1  6 18 13; 2  7 23 18]);
   unit_test_cmp('Sentec',fmdl.normalize_measurements, false);

   opt.skip=2;
   fmdl = fwd_model_device(fmdl,'Sentec',opt);
   ssmm = stim_meas_list(fmdl.stimulation);
   unit_test_cmp('Sentec',size(ssmm,1),736);
   unit_test_cmp('Sentec',ssmm([8,35],:), [
       1  4 16 13; 2  5 21 18]);

   opt.next=2;
   fmdl = fwd_model_device(fmdl,'Sentec',opt);
   ssmm = stim_meas_list(fmdl.stimulation);
   unit_test_cmp('Sentec',size(ssmm,1),672);
   unit_test_cmp('Sentec',ssmm([8,35],:), [
       1  4 17 14; 2  5 24 21]);
