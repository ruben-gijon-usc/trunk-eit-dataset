function vfilt = freq_filt(vin, fresp, FR, dim)
%FREQ_FILT: frequency filter data
% vfilt = freq_filt(vin, fresp, FR, dim)
% vin = matrix of data values
%     = data structure (with field vin.meas)
%     = image structure (with field vin.elem_data)
% fresp = function of freq filter
%   e.g. fresp = @(f) f<10;  % values in Hz
%   e.g. fresp = @(f) (1+(f/10).^6).^(-0.5);  % Butter 2*3rd order
% FR  = frame rate (or use FR parameter on data structures)
%    or FR is a structure with fields
%     .FR = Frame Rate
%     .padding = length of padding (in seconds)
% dim = dimension along which to filter (default is 2)
%     .padding_type = zero|extend|linear
%       pad with zeros, extend the last values (default), or linearly fit between start/end

% (C) Andy Adler 2019. License: GPL v2 or v3.
% $Id: freq_filt.m 6977 2024-11-03 15:23:53Z aadler $

% if FR not given, see if it's a parameter

if ischar(vin) && strcmp(vin,'UNIT_TEST'); do_unit_test; return; end

p.padding = 30; % seconds default;
p.padding_type = 'extend';
if nargin<3; 
   p.FR = vin.FR;
else
   if isnumeric(FR)
     p.FR = FR;
   elseif isstruct(FR);
     for ff= fieldnames(FR)'; % wish matlab could do this easily
       p.(ff{1}) = FR.(ff{1});
     end
   else
     error('Don''t understand parameter FR');
   end
     
      
end
if nargin>=4
   p.dim = dim;
elseif ~isfield(p,'dim');
   p.dim = 2;
end

if isstruct(vin) && isfield(vin,'type');
   switch vin.type
     case 'data'
        vin.meas = do_freq_filt(vin.meas, fresp, p);
        vfilt = vin;
     case 'image'
        vin.elem_data = do_freq_filt(vin.elem_data, fresp, p);
        vfilt = vin;
     otherwise
        error('Can''t process object of type %',vin.type);
   end
elseif isnumeric(vin)
   vfilt = do_freq_filt(vin, fresp, p);
else 
   error('can''t process object');
end
      


% Filter in the frequency direction
function s = do_freq_filt(s,fresp, p)
  rs = isreal(s);
  s = padsignal(s,p);
  f = fft(s,[],p.dim);
  fax = freq_axis_filt(p.FR,size(s,p.dim),fresp);
  f = f .* fshape(p,fax);
  s= ifft(f,[],p.dim);
  if rs % is signal is real
     if norm(imag(s(:))) / norm(real(s(:))) > 1e-12
        error('FFT filter has imag output');
     end
     s = real(s);
  end
  s = unpadsignal(s,p);

function s = fshape(p,s);
  fsh = ones(1,length(size(s)));
  fsh(p.dim) = prod(size(s));
  s= reshape(s,fsh);

function plen = pad_len(p);
  plen = round(p.FR * p.padding);

% Add padding connecting the last to the first sample
function s = padsignal(s,p);
  lsup = linspace(0,1,pad_len(p)); lsup=fshape(p,lsup);
  lsdn = linspace(1,0,pad_len(p)); lsdn=fshape(p,lsdn);
  switch p.padding_type
    case 'zero';
      pad= calc_padding(s,0*lsdn,0*lsup,p);
    case 'extend';
      pad= calc_padding(s,lsdn,lsup,p);
    case 'linear';
      mn  = mean(s,p.dim); ss = s-mn;
      xx  = linspace(-1,1,size(ss,p.dim));
      perm = ones(1,5); perm(p.dim) = length(xx); % do we ever want d>5??
      xx  = reshape(xx,perm);
      a   = mean(ss.*xx,p.dim) ./mean(xx.^2,p.dim);
      pad = (mn-a).*lsdn + (mn+a).*lsup;
    otherwise; 
      error('padding type not understood');
  end
  
  s = cat(p.dim,s,pad);

function pad= calc_padding(s,lsdn,lsup,p)
  switch p.dim % wish I knew how to generalize in m*lab
    case 1; pad = s(1,:,:).*lsdn + s(end,:,:).*lsup;
    case 2; pad = s(:,1,:).*lsdn + s(:,end,:).*lsup;
    case 3; pad = s(:,:,1).*lsdn + s(:,:,end).*lsup;
    otherwise; error('Problem with dim above 3');
  end

% Add padding connecting the last to the first sample
function s = unpadsignal(s,p);
  plen = round(p.FR * p.padding);
  switch p.dim % wish I knew how to generalize in m*lab
    case 1; s(end+1-(1:plen),:,:) = [];
    case 2; s(:,end+1-(1:plen),:) = [];
    case 3; s(:,:,end+1-(1:plen)) = [];
    otherwise; error('Problem with dim above 3');
  end
  
function fax = freq_axis_filt( FR, lD, fresp);
  fax = linspace(0,FR,lD+1);
  fax(end)=[];
  fax(fax>FR/2) = fax(fax>FR/2) - FR;
  fax = feval(fresp, abs(fax));

function do_unit_test
  clf; subplot(3,1,1);
  FR = 100;
  t = (0:1.1e3)/FR;
  s = sin(2*pi*5*t) + 2*cos(2*pi*15*t) +  3*sin(2*pi*0.1*t) + 1;
  subplot(211); plot(t,s); hold on;
  subplot(212); plot(t,s); hold on; xlim(max(t)-[0.5,0]);
  pp.FR = FR;
  pp.padding_type = 'extend';
  pp.padding = 1;

  fnum=0; while true; fnum=fnum+1; switch fnum
     case 1; fresp = @(f) f<10;
             sf= freq_filt(s,fresp, pp);
             sf12=[1.879618111164496, 1.275026363303550];

     case 2; fresp = @(f) (f<1) + (f>=1)./(f+eps);
             p.padding_type = 'extend';
             p.FR = FR; p.padding = 1;
             sf= freq_filt(s,fresp, p);
             sf12=[2.033878107482741, 1.790784258437183];
     case 3; fresp = @(f) f<1;
             p.padding_type = 'extend';
             p.FR = FR; p.padding = 0;
             sf= freq_filt(s,fresp, p);
             sf12=[0.930430975008685, 0.910716555382052];
     case 4; fresp = @(f) f<1;
             p.padding_type = 'extend';
             p.FR = FR; p.padding = 2;
             sf= freq_filt(s,fresp, p);
             sf12=[1.977191625314283, 1.912923819876781];
     case 5; fresp = @(f) f<1;
             p.padding_type = 'linear';
% TODO- debug for padding size differences
             p.FR = FR; p.padding = 3;
             sf= freq_filt(s,fresp, p);
             sf12=[1.231446973826249, 1.277407267967623]-2;
     case 6; fresp = @(f) f<1;
             p.padding_type = 'zero';
% TODO- debug for padding size differences
             p.FR = FR; p.padding = 1;
             sf= freq_filt(s,fresp, p);
             sf12=[1.818358872667770, 1.846165834456450]-2;
     case 7; fresp = @(f) f<10;
             sf= freq_filt(s,fresp, FR);
             sf12=[1.884878325803839, 1.271527154193287];
     otherwise; break
     end
     unit_test_cmp(sprintf('ff%02d',fnum),sf(1:2)-1, sf12,1e-13)
     subplot(211); plot(t,sf);
     subplot(212); plot(t,sf);
  end
  legend('0','1','2','3','4','5','6');
  hold off;

  sv = s.*[2;1;3;4];
  fresp = @(f) f<10;
  sf= freq_filt(sv,fresp, pp);
  sf12=[1.879618111164496, 1.275026363303550]+1;
  unit_test_cmp('ff10',sf(2,1:2), sf12,1e-13)

  sf= freq_filt(sv,fresp, pp, 2);
  unit_test_cmp('ff11',sf(2,1:2), sf12,1e-13)

  sf= freq_filt(sv',fresp, pp, 1);
  unit_test_cmp('ff12',sf(1:2,2), sf12',1e-13)

  so = struct('type','data','meas',sv);
  sf= freq_filt(so,fresp, pp, 2);
  unit_test_cmp('ff21',sf.meas(2,1:2), sf12,1e-13)
  
  so = struct('type','image','elem_data',sv);
  sf= freq_filt(so,fresp, pp, 2);
  unit_test_cmp('ff21',sf.elem_data(2,1:2), sf12,1e-13)

  % TODO add test for complex input
