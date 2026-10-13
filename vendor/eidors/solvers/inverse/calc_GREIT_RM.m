function [RM, PJt, M, noiselev] = calc_GREIT_RM(vh,vi, xyc, radius, weight, options)
% CALCULATE GREIT reconstruction matrix
%   RM= calc_GREIT_RM(vh,vi, xyc, radius, weight, normalize)
% 
% Input:
%   vh     = homogeneous (reference) training measurements 
%   vi     = inhomogeneous training measurements 
%   xyc    = x,y position of targets 2xN
%   radius = requested resolution matrix
%      if scalar - apply resolution to all images:  recommend 0.25 for 16 elecs
%   weight = weighting matrix
%      if scalar   = weighting of noise vs signal
%      if 32^2 x N = weighting of each image output
%   options.normalize = 
%            0 -> regular difference EIT
%            1 -> normalized difference EIT
%   options.meshsz = [xmin xmax ymin ymax] size of mesh
%      defaults to [-1 1 -1 1]
%   options.imgsz  = [xsz ysz] size of the reconstructed image in pixels
%   options.noise_covar [optional]
%      covariance matrix of data noise
%      - use to specify electrode errors
%      - use to add movement (Jm * Jm')
%   options.desired_solution_fn
%      specify a function to calculate the desired image. 
%      It must have the signature:
%      D = my_function( xyc, radius, options);
%      uses eidors_defualt('get','GREIT_desired_img') if not specified
% 
% See also: mk_GREIT_model 
%
% (C) 2009 Andy Adler. Licenced under GPL v2 or v3
% $Id: calc_GREIT_RM.m 6681 2024-03-14 19:37:03Z aadler $

   if ischar(vh) && strcmp(vh,'UNIT_TEST'); do_unit_test; return; end

   if ~isstruct(options)
       options.normalize = options;
   end
   opt = parse_options(options);

   if opt.normalize
      Y = vi./(vh*ones(1,size(vi,2))) - 1; 
   else
      Y = vi - (vh*ones(1,size(vi,2)));
   end
   try 
       f = opt.desired_solution_fn;
   catch
       f = eidors_default('get','GREIT_desired_img');
   end

   D = feval(f, xyc, radius, opt);
   
   % PJt is expensive and doesn't change when optimising NF
   copt.cache_obj = {vi,vh,xyc,radius,opt};
   copt.fstr = 'calc_GREIT_RM_PJt';
   PJt = eidors_cache(@calc_PJt,{Y,D},copt);
   
   if isscalar(weight)
       [RM, M, noiselev] = calc_RM(PJt, Y, weight, opt);
   else
       error('use of weight matrix is not yet available');
   end

function [RM, M, noiselev] = calc_RM(PJt, Y, noiselev, opt)

   noiselev = noiselev * mean(abs(Y(:)));
   % Desired soln for noise is 0
   N_meas = size(Y,1);

   % This implements RM = D*Y'/(J*Sx*J + Sn);
   Sn = speye(N_meas) * opt.noise_covar; % Noise covariance
%    PJt= D*Y';
   % Conjugate transpose here
   M  = (Y*Y' + noiselev^2*Sn);
   % Ensure to use transpose not conjugate'
   RM =  left_divide(M.',PJt.').';    %PJt/M;

function [RM, M] = calc_RM_testcode(PJt, Y, noiselev, opt)

   noiselev = noiselev * mean(abs(Y(:)));
   % Desired soln for noise is 0
   N_meas = size(Y,1);

   % This implements RM = D*Y'/(J*Sx*J + Sn);
   Sn = speye(N_meas) * opt.noise_covar; % Noise covariance
%    PJt= D*Y';
   % Conjugate transpose here
   M  = (Y*Y' + noiselev^2*Sn);
   % OLD and TESTCODE
   Y = [Y, noiselev*eye(N_meas)];

   RM = PJt/(Y*Y');

function PJt = calc_PJt(Y,D)
   PJt = D*Y';
      

function opt = parse_options(opt)
   if ~isfield(opt, 'normalize'), opt.normalize = 1; end
   if ~isfield(opt, 'meshsz'),    opt.meshsz = [-1 1 -1 1]; end
   if ~isfield(opt, 'imgsz'),     opt.imgsz = [32 32]; end
   if ~isfield(opt, 'noise_covar'),
                 opt.noise_covar = 1;
   end
%  options.data_covariance [optional]

function do_unit_test
   img= mk_image(mk_common_model('a2c2',16));
   J = calc_jacobian(img);
   opt.noise_covar = 1;
   noiselev = 1;
   [RM, M] = calc_RM(J', J, noiselev, opt);
   img.elem_data = RM*J(:,20);
   show_fem(img); 
   unit_test_cmp('a2c2 noiselev=1', img.elem_data(19:21), ...
  [ -0.108227622116164; 0.737404299070634; -0.009842231726770], 1e-12);

   [RMold, M] = calc_RM_testcode(J', J, noiselev, opt);
   unit_test_cmp('test RM calc', RM, RMold, 1e-10);
