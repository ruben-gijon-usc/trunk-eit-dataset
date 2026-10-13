function N = num_lights(ax)
%NUM_LIGHTS return the number of light objects in an axes
%
% NUM_LIGHTS() uses the current axes
%
% NUM_LIGHTS(AX) uses specified axes AX
%
% See also: LIGHT, CAMLIGHT



% (C) 2024 Bartek Grychtol.
% License: GPL version 2 or version 3
% $Id: num_lights.m 7124 2024-12-29 15:18:32Z aadler $

if nargin == 1 && ischar(ax) && strcmp(ax,'UNIT_TEST'), do_unit_test; return,end

if nargin < 1
   ax = gca;
end
ch = get(ax, 'Children');


ver= eidors_obj('interpreter_version');
if ver.isoctave
   % children are scalar
   N = sum(strcmp(get(ch, 'type'), 'light'));
else
   N = 0;
   for i = 1:numel(ch)
      N = N + isa(ch(i),'matlab.graphics.primitive.Light');
   end
end

function do_unit_test
  clf
  subplot(211)
  plot(rand(5));
  camlight('left')
  camlight('right')
  ax = gca;
  unit_test_cmp('current axis', num_lights, 2);
  subplot(212)
  unit_test_cmp('specific axis', num_lights(ax), 2);

