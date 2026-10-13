function mdl = set_predef_stim(mdl, stimpat)
% SET_PREDEF_STIM 
% Common helper function to configure electrodes and stim pattern on thorax
% models.
%
% See also: MK_THORAX_MODEL_GRYCHTOL2016a, MK_THORAX_MODEL_BP3D

% (C) 2016-2024 Bartlomiej Grychtol. License: GPL version 2 or 3
% $Id: set_predef_stim.m 7002 2024-11-24 13:11:35Z aadler $

if strcmp(mdl.type, 'image')
   img = mdl;
   mdl = img.fwd_model;
end

emap = get_elec_map(stimpat);
mdl.electrode = mdl.electrode(emap);
stim = get_stim_pattern(stimpat);
mdl = eidors_obj('set', mdl, 'stimulation', stim); % standard field order

if exist('img','var')
   img.fwd_model = mdl;
   mdl = img;
end


end

function stim = get_stim_pattern(str)
   switch(str)
      case '2x16_planar'
         stim = mk_stim_patterns(32,1,[0 6],[0 6],{'no_meas_current','no_rotate_meas'},1);
      case {'2x16_odd-even', '2x16_square', '1x32_ring'}
         stim = mk_stim_patterns(32,1,[0 5],[0 5],{'no_meas_current','no_rotate_meas'},1);
      case '2x16_adjacent'
         stim = mk_stim_patterns(32,1,[0 1],[0 1],{'no_meas_current','no_rotate_meas'},1);  
      otherwise
         error('Stim pattern string not understood. Available strings are: \n%s', ...
            sprintf('%s\n', pattern_list));
   end
end


function ls = pattern_list
   ls = {
      '2x16_planar'
      '2x16_odd-even'
      '2x16_square'
      '2x16_adjacent'
      '1x32_ring'
      };
end

function map = get_elec_map(str)

   switch str
      case {'2x16_odd-even', '2x16_planar'}
         map = oddeven32;
      case {'2x16_square', '2x16_adjacent'}
         map = square32;
      case '1x32_ring'
         map = ring32;
      otherwise
         error('No such electrode map');
   end
end

function m = square32
   o = [48 1 -48 1];
   o = repmat(o,1,8);
   m = zeros(1,32);
   m(1) = 1;
   for i = 2:32
      m(i) = m(i-1) + o(i-1);
   end
end


function m = oddeven32
   odd = 49:64; % top layer
   even = 1:16; % bottom layer
   m = [odd; even];
   m = m(:)';
end

function m = ring32
   m = 17:48;
end