function hh=vertlines( xpts, opts );
% VERTLINES: put vertical lines at time markers
%   vertlines( time_pts )
%   vertlines( time_pts, opts ) % line opts i.e. {'LineWidth',2}

% $Id: vertlines.m 7089 2024-12-20 20:24:40Z aadler $ (C) A.Adler 2024. Copyright GPL v2 or v3

  if nargin==1; opts = {}; end
  if ischar(xpts) && strcmp(xpts,'UNIT_TEST'); do_unit_test; return; end

  xpts = reshape(xpts,1,[]);
  oo   = 1+0*xpts;
  yl=ylim;
  % -negative z to put at back
  hh=line([1;1]*xpts,yl'*oo,-[1;1]*oo,'Color',[0,0,0], ...
      'HandleVisibility','off',opts{:});
  ylim(yl);


function do_unit_test
  t=linspace(0,100,1000);
  plot(t,sin(t).*exp(-t/100));
  vertlines([0:20:60]);
  vertlines([50],{'LineWidth',3,'Color',[0,1,0]});
