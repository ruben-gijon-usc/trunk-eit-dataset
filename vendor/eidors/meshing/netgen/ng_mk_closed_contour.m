function [fmdl,mat_idx] = ng_mk_ellip_models(ellip_shape, elec_pos, ...
                  elec_shape, extra_ng_code);
% NG_MK_CLOSED_CONTOUR: fit elliptical model to a contour
%[fmdl,mat_idx] = ng_mk_closed_contour(shape, elec_pos, ...
%                 elec_shape, extra_ng_code);
%
% This function creates netgen models of a closed contour in x,y.
%   An elliptical FEM is created and then perturbed using Fourier 
%   descriptors into the final model.
% It should work well for cases where the contour is roughtly elliptical
%
% INPUT:
% ellip_shape = cell{height, xy_points, [maxsz]]}
%    if height = 0 -> calculate a 2D shape
%    xy_points     -> Npoints x 2. points on the object boundary
%    maxsz  (OPT)  -> max size of mesh elems (default = course mesh)
%
% ELECTRODE POSITIONS:
%  elec_pos = [n_elecs_per_plane,z_planes] 
%     OR
%  elec_pos = [degrees,z] centres of each electrode (N_elecs x 2)
% Note that electrode positions are in the ellipse before fitting
%
% ELECTRODE SHAPES::
%  elec_shape = [width,height, maxsz]  % Rectangular elecs
%     OR
%  elec_shape = [radius, 0, maxsz ]    % Circular elecs
%     OR 
%  elec_shape = [0, 0, maxsz ]         % Point electrodes
%    (point elecs does some tricks with netgen, so the elecs aren't exactly where you ask)
%
% Specify either a common electrode shape or for each electrode
%
% EXTRA_NG_CODE
%   string of extra code to put into netgen geo file. Normally this
%   would be to insert extra materials into the space
%
% OUTPUT:
%  fmdl    - fwd_model object
%  mat_idx - indices of materials (if extra_ng_code is used)
%    Note mat_idx does not work in 2D. Netgen does not provide it.
%
%
% USAGE EXAMPLES:
% Thorax shape
% xy_points = .01*[
%  963 534;1013 538;1056 535;1102 517;1130 492;
% 1150 454;1169 419;1182 375;1172 345;1149 318;
% 1112 301;1095 297;1039 285; 982 284; 928 283;
%  870 279; 816 299; 776 328; 761 374; 782 429;
%  805 482; 844 518; 900 533; 961 532];
% % refine cylendar from ellip_shape
% ng_write_opt('MSZCYLINDER',[0,0,1,0,0,1.2,2,.05]);
%  fmdl= ng_mk_closed_contour({3,xy_points},[8,1.1,1.9],[0.1]); 
% 

% (C) Andy Adler, 2010. Licenced under GPL v2 or v3
% $Id: ng_mk_closed_contour.m 6934 2024-06-12 20:27:34Z aadler $

if ischar(ellip_shape) && strcmp(ellip_shape,'UNIT_TEST'); do_unit_test; return; end

if nargin < 4; extra_ng_code = {'',''}; end
% Caching should be done in the ng_mk_ellip_models

[ellip_shape,params] = analyse_shape(ellip_shape);
[fmdl, mat_idx] = ng_mk_ellip_models(ellip_shape, elec_pos, ...
                  elec_shape, extra_ng_code);
fmdl = fit_to_shape( fmdl, params );

function [ellip_shape,params] = analyse_shape(ellip_shape);
  height = ellip_shape{1}; 
  xy_shape=ellip_shape{2};
  if length(ellip_shape)>=3
    maxh= ellip_shape{3};
  else
    maxh= [];
  end

  [u,d,v]= svd(cov(xy_shape));
  params.rot = [1,0;0,-1]*u;
  params.xy_mean = mean(xy_shape,1);

  ellip_ax = sqrt(2*[d(1,1),d(2,2)]);
  ellip_shape= [height, ellip_ax, maxh];

  N = min(15, ceil( size(xy_shape,1) / 3 )); % can't fit too many components
  params.f_fit_point = fourier_fit(xy_shape, N);

% Now, fit the ellipse to the point
  th = linspace(0,2*pi,4*N)';
  ellip_pts = [cos(th),sin(th)]*sqrt(2*d)*params.rot;
  params.f_fit_ellip = fourier_fit(ellip_pts, N);
%plot(ellip_pts(:,1),ellip_pts(:,2),'b-*',xy_shape(:,1),xy_shape(:,2),'r-*')

function C = fourier_fit(xy_shape,N);
% Fourier Fit
% [1, cos(t1), cos(2*t1), ... sin(t1) ...] * [C1] = [R1]
%            ...                         
% [1, cos(tN), cos(2*tN), ... sin(tN) ...] * [CN] = [RM]

   xc = xy_shape(:,1); xm= mean(xc);   xc= xc-xm;
   yc = xy_shape(:,2); ym= mean(yc);   yc= yc-ym;

   ang = atan2(yc,xc);
   rad = sqrt(yc.^2 + xc.^2);
   A = ones(length(ang), 2*N+1);
   for i=1:N
     A(:,i+1  ) = cos(i*ang);
     A(:,i+1+N) = sin(i*ang);
   end
   C = A\rad;

function fmdl = fit_to_shape( fmdl, params );
   % Rotate and move the objects
   xnodes = fmdl.nodes(:,1:2)*params.rot(:,1);
   ynodes = fmdl.nodes(:,1:2)*params.rot(:,2);

   N = (length(params.f_fit_point)-1)/2;
   [ang,rad] = cart2pol(xnodes,ynodes);
   A = ones(length(ang), 2*N+1);
   for i=1:N
     A(:,i+1  ) = cos(i*ang);
     A(:,i+1+N) = sin(i*ang);
   end

   r_ellip = A*params.f_fit_ellip;
   r_point = A*params.f_fit_point;
   rad = rad.* (r_point ./ r_ellip);
   xnodes = rad.*cos(ang) + params.xy_mean(1);
   ynodes = rad.*sin(ang) + params.xy_mean(2);

   fmdl.nodes(:,1) = xnodes;
   fmdl.nodes(:,2) = ynodes;

function do_unit_test
   xy_points = [
    963 534; 987 535;1013 538;1036 540;1056 535;
   1077 527;1102 517;1115 506;1130 492;1141 474;
   1150 454;1157 435;1169 419;1179 397;1182 375;
   1178 358;1172 345;1163 331;1149 318;1132 307;
   1112 301;1097 293;1095 297;1077 289;1039 285;
   1011 284; 982 284; 955 285; 928 283; 900 280;
    870 279; 845 289; 816 299; 788 310; 776 328;
    767 349; 761 374; 768 395; 782 429; 795 456;
    805 482; 823 505; 844 518; 868 523; 900 533;
    929 536; 961 532]/100;
   fmdl= ng_mk_closed_contour({3,xy_points},[8,1.1,1.9],[0.1]); 
   show_fem(fmdl)
   unit_test_cmp('contour A',num_elecs(fmdl),16);
   mnod = max(fmdl.nodes,[],1); 
   unit_test_cmp('contour B',mnod(3),3)
   unit_test_cmp('contour C',mnod(1:2),max(xy_points),3e-2)
   mnod = min(fmdl.nodes,[],1); 
   unit_test_cmp('contour D',mnod(3),0)
   unit_test_cmp('contour E',mnod(1:2),min(xy_points),3e-2)

