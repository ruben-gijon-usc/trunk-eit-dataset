function Reg= prior_gaussian_HPF( fwd_model );
% PRIOR_GAUSSIAN_HPF calculate image prior
% Reg= prior_gaussian_HPF( fwd_model )
%     or accepts inv_model
% Reg        => output regularization term
% fwd_model  => forward model struct
% Parameters:
%   diam_frac= fwd_model.prior_gaussian_HPF.diam_frac DEFAULT 0.1
%   zero_thresh=fwd_model.prior_gaussian_HPF.zero_thresh DEFAULT 0.0001
%
%
% prior_gaussian_HPF is designed to be used as an R_prior, rather than a RtR_prior
%
% CITATION_REQUEST:
% AUTHOR: A Adler & R Guardo
% YEAR: 1996
% TITLE: Electrical impedance tomography: regularized imaging and contrast
% detection 
% JOURNAL: IEEE transactions on medical imaging
% VOL: 15
% NUM: 2
% PAGE: 170–9
% LINK: http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=491418

% NOTES:
% - the old version had numerous bugs. It has been fixed and simplified for version 3.11
% - Old code should be removed in future
% TODO:
% - Accept and work with coarse2fine

% (C) 2005 Andy Adler. License: GPL version 2 or version 3
% $Id: prior_gaussian_HPF.m 6510 2022-12-30 16:34:22Z aadler $

if ischar(fwd_model) && strcmp(fwd_model,'UNIT_TEST'); do_unit_test; return; end

if ~strcmp(fwd_model.type, 'fwd_model')
    fwd_model= fwd_model.fwd_model;
end
try 
    diam_frac= fwd_model.prior_gaussian_HPF.diam_frac;
catch
    diam_frac= 0.1;
end
try 
    diam_frac= fwd_model.prior_gaussian_HPF.zero_thresh;
catch
    zero_thresh = .0001;
end

copt.cache_obj= {fwd_model.nodes, fwd_model.elems, diam_frac};
copt.fstr = 'prior_gaussian_HPF';
Reg = eidors_cache(@calc_Gaussian_HPF, {fwd_model, diam_frac, zero_thresh}, copt );

function filt= calc_Gaussian_HPF( fmdl, diam_frac, zero_thresh)
  pts = interp_mesh(fmdl,3); % elem ctr
  fmdl.interp_mesh.n_interp = 0; % for ctr
  ptc = interp_mesh(fmdl,0);
  % we divide by four to closely match
  % previous implementation
  mdl_dim = max(fmdl.nodes) - min(fmdl.nodes);
  mdl_dim = mean(mdl_dim(1:2)); 
   
  beta=2.769/(diam_frac * mdl_dim)^2;
  dim = elem_dim(fmdl);

  % Gaussian is integral of exp(-beta*r^2)
  % Integral of exp(-1/2 * x'*Sigma*x)
  %  = sqrt((2*pi)^k*det(Sigma))
  % Here Sigma = 2*eye()*beta
  %  det(Sigma) = (2*beta)^k
  % so int[exp(-beta*r2)]= (pi*beta)^(d/2)

  % integral of exp(-r2) = 
  Abeta_pi = get_elem_volume(fmdl,'no_c2f') ...
             *(beta/pi)^(dim/2);
  for j= 1:num_elems(fmdl)
    dif = bsxfun(@minus, pts, ptc(j,:));
    r2  = sum(dif.^2, 2);
    mean_elem = mean( exp(-beta*r2 ),3);
    flt(:,j)=Abeta_pi.*mean_elem;
  end %for j=1:ELEM
  % flt should sum to 1, but the integral is numeric and could vary. Instead we normalize
  flt = flt./sum(flt);
  % Filter is 1 - Gaussian
  flt=eye(num_elems(fmdl)) - flt;
  flt= ( flt+flt' )/ 2;
  filt= sparse(flt.*(abs(flt)>zero_thresh)); 

% Calculate Gaussian HP Filter as per Adler & Guardo 96
% parameter is diam_frac (normally 0.1)
function filt= calc_Gaussian_HPF_old( fmdl, diam_frac)
  ELEM= fmdl.elems';
  NODE= fmdl.nodes';


  e= size(ELEM, 2);
  np= 128;
  [x,xc,y,yc] = interp_points(NODE,ELEM,np);

  v_yx= [-y,x];
  o= ones(np*np,1);
  filt= zeros(e);
  tourne= [0 -1 1;1 0 -1;-1 1 0];

  for j= 1:e
%   if ~rem(j,20); fprintf('.'); end
    xy= NODE(:,ELEM(:,j))';
    a= xy([2;3;1],1).*xy([3;1;2],2) ...
         -xy([3;1;2],1).*xy([2;3;1],2);
    aire=abs(sum(a));
    mx_xy = max(xy);
    mn_xy = min(xy);
    endr=find(y<=mx_xy(2) & y>=mn_xy(2) ...
            & x<=mx_xy(1) & x>=mn_xy(1) )';
    aa= sum(abs(ones(length(endr),1)*a' ...
         +v_yx(endr,:)*xy'*tourne)');
    endr( find( abs(1 - aa / aire) > 1e-8 ) )=[];
    ll=length(endr); endr=endr-1;

    yp= rem(endr,np)/(np-1) - .5; % (rem(endr,np) corresponde a y
    ym= ones(e,1)*yp -yc*ones(1,ll);
    xp= floor(endr/np)/(np-1) - .5; % (floor(endr/np)) corresponde a x
    xm= ones(e,1)*xp -xc*ones(1,ll);

    beta=2.769/diam_frac.^2;
%   filt(:,j)=-aire/2*beta/pi*mean(...
    filt(:,j)=-beta/pi*sum( exp(-beta*(ym.^2+xm.^2))')';
  end %for j=1:ELEM
% filt=filt/taille(1)/taille(2)+eye(e);
  filt=filt/np^2+eye(e);
  filt= ( filt+filt' )/ 2;
  filt= sparse(filt.*(abs(filt)>.003)); 

function [x,xc,y,yc] = interp_points(NODE,ELEM,np);
  taille=max(NODE')-min(NODE');
  e= size(ELEM, 2);

% Triangles of each shape
  xt= reshape(NODE(1,ELEM(:)),3,e)';
  yt= reshape(NODE(2,ELEM(:)),3,e)';

% We want center [1,1,1]/3 and edges [4,1,1]/6
  pts= [2,2,2;4,1,1;1,4,1;1,1,4]'/6;
  xp= xt*pts;
  yp= yt*pts;
  
  [x y]=meshgrid( ...
      linspace( min(NODE(1,:)), max(NODE(1,:)) ,np ), ...
      linspace( min(NODE(2,:)), max(NODE(2,:)) ,np )  ); 
% Add the basic interpolation points to those based on the
%  elements
  x= [x(:);xp(:)]; 
  y= [y(:);yp(:)]; 

% BUG HERE: why is xc,yc scaled and not x,y

  xc= mean(xt,2)/taille(1);
  yc= mean(yt,2)/taille(2);

function do_unit_test
  imdl = mk_common_model('a2c0',16);
  RtR = prior_gaussian_HPF(imdl);
  tt=[0.557367798877852, -0.124016055611277, -0.029231532288002, -0.124016055611277];
  unit_test_cmp('a2c2 :1', RtR(1,1:4),tt,1e-10);

  % Old values are somewhat similar
  RtR= calc_Gaussian_HPF_old( imdl.fwd_model, 0.1);
  tt=[0.562239752317943, -0.117068756722254, -0.025875127622824, -0.117068756722254];
  unit_test_cmp('a2c2 _old :1', RtR(1,1:4),tt,1e-10);

  imdl = mk_common_model('a3cr',16);
  RtR = prior_gaussian_HPF(imdl);  %NOTE: Fix required
  tt=[0.793660877689329,-0.096517734290331;
     -0.096517734290331, 0.793660877689329;
     -0.004236643618993,-0.011516904603663;
     -0.011516904603663,-0.004236643618993];
  unit_test_cmp('a3cr :1', RtR(1:4,1:2),tt,1e-10);

  ls = -1:.1:1;
  fmdl = mk_grid_model([], ls, ls); 
  fmdl = rmfield(fmdl,'coarse2fine');
  RtR = prior_gaussian_HPF(fmdl);
  
  % Test space-independence ... 2D
  ll= 2*(length(ls)-1);
  unit_test_cmp('2D space indep', ...
     RtR(10*ll+(1:4),10*ll+(1:4)), ...
     RtR(12*ll+(1:4),12*ll+(1:4)), 1e-14);


  ls = -1:.2:1;
  fmdl = mk_grid_model([],ls,ls,(-4:2:4)/10); 
  fmdl = rmfield(fmdl,'coarse2fine');
  RtR = prior_gaussian_HPF(fmdl);
  
  % Test space-independence ... 2D
  ll= 6*(length(ls)-1);
  lv= 6*(length(ls)-1)^2;
  unit_test_cmp('3D space indep', ...
     RtR(lv+3*ll+(1:4),lv+3*ll+(1:4)), ...
     RtR(lv+5*ll+(1:4),lv+5*ll+(1:4)), 1e-14);

