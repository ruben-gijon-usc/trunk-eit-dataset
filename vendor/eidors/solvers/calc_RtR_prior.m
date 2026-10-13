function RtR_prior = calc_RtR_prior( inv_model )
% RtR = calc_RtR_prior( inv_model )
% CALC_RtR_PRIOR: calculate image regularization prior
%   R'*R (which is an estimate of the inverse of the covariance)
%
%   Typically, the image prior is matrix n_elem x n_elem of the
%   normalized a priori crosscorrelation FEM element values
% 
% calc_RtR_prior can be called as
%    RtR_prior= calc_RtR_prior( inv_model, ... )
%
% and will call the function inv_model.RtR_prior
% parameters to RtR_prior should be passed in the field
% inv_model.RtR_prior_function_name.parameters
%
% If inv_model.RtR_prior is a matrix, calc_RtR_prior will return that matrix,
% possibly correcting for coarse2fine
%
% if there exists a field inv_model.rec_model, then
%   the prior is calculated on the rec_model rather than
%   the fwd_model. This will not be done if 
% inv_model.prior_use_fwd_not_rec= 1;
%
% RtR_prior    the calculated RtR regularization prior
% inv_model    is an inv_model structure
%
% If a function to calculate RtR_prior is not provided,
% RtR = R_prior' * R_prior;

% (C) 2005-2018 Andy Adler. License: GPL version 2 or version 3
% $Id: calc_RtR_prior.m 7097 2024-12-23 16:17:37Z aadler $

if ischar(inv_model) && strcmp(inv_model,'UNIT_TEST'); do_unit_test; return; end

inv_model = rec_or_fwd_model( inv_model);

if isfield(inv_model,'RtR_prior')
   if isnumeric(inv_model.RtR_prior)
      RtR_prior = inv_model.RtR_prior;
   else
      try inv_model.RtR_prior = str2func(inv_model.RtR_prior); end
      RtR_prior= feval( inv_model.RtR_prior, inv_model );
   end
   % cache results of c2f call
   RtR_prior= eidors_cache(@c2f_RtR_prior, ...
         {inv_model,RtR_prior}, 'calc_RtR_prior' );
elseif isfield(inv_model,'R_prior')
   RtR_prior = eidors_cache(@calc_from_R_prior, inv_model, 'calc_RtR_prior');
else
   error('calc_RtR_prior: neither R_prior nor RtR_prior provided');
end

function RtR_prior = c2f_RtR_prior( inv_model, RtR_prior);
   if isfield(inv_model.fwd_model,'coarse2fine')
      c2f= inv_model.fwd_model.coarse2fine;
      if size(RtR_prior,1)==size(c2f,1)
   %     we need to take into account coarse2fine - using a reasonable tol
         eidors_msg('calc_RtR_prior: using coarse2fine to model RtR');
         f2c= c2f'; %pinv(c2f,1e-6);
         RtR_prior=f2c*RtR_prior*c2f;

         % now update the object in the cache
      end
   end

function RtR_prior = calc_from_R_prior(inv_model)

   % The user has provided an R prior. We can use this to
   % calculate RtR= R'*R;
   if isnumeric(inv_model.R_prior)
      R = inv_model.R_prior;
   else
      try inv_model.R_prior = str2func(inv_model.R_prior); end
      R= eidors_cache( inv_model.R_prior, inv_model );
   end

   RtR_prior = c2f_RtR_prior(inv_model, R'*R );

   
   

function inv_model = rec_or_fwd_model( inv_model);

   if isfield(inv_model,'rec_model');
      use_rec_model = 1;
      try if inv_model.prior_use_fwd_not_rec== 1;
         use_rec_model = 0;
      end; end

      if use_rec_model
          % copy the normalize flag from the fwd_model to prevent warnings
         inv_model.rec_model = mdl_normalize(inv_model.rec_model, ...
                                       mdl_normalize(inv_model.fwd_model));
         inv_model.fwd_model= inv_model.rec_model;
         inv_model= rmfield(inv_model,'rec_model');
      end
   end


function do_unit_test
   imdl = mk_common_model('a2c2',16);;
   imdl.RtR_prior = @prior_tikhonov;
   RtR = calc_RtR_prior(imdl);
   unit_test_cmp('RtR=prior_tik',RtR,speye(size(RtR,1)),1e-14);
   imdl.RtR_prior = @prior_laplace;
   RtR = calc_RtR_prior(imdl);
   unit_test_cmp('RtR=prior_lapl',RtR(1:3,1:3),[6,-2,0;-2,6,-2;0,-2,6],1e-14);

   imdl = rmfield(imdl,'RtR_prior');
   imdl.R_prior = @prior_tikhonov;
   RtR = calc_RtR_prior(imdl);
   unit_test_cmp('R=prior_tik',RtR,speye(size(RtR,1)),1e-14);
   imdl.R_prior = @prior_TV;
   RtR = calc_RtR_prior(imdl);
   unit_test_cmp('R=prior_TV',RtR(1:3,1:3)*16,[4,-1,0;-1,4,-1;0,-1,4],1e-14);

   imdl = mk_common_model('b2c2',16);
   grid = linspace(-1,1,16);
   [imdl.rec_model, imdl.fwd_model.coarse2fine] = ...
        mk_grid_model( imdl.fwd_model, grid, grid);
   RtR = calc_RtR_prior(imdl);
   disp('cache should only be called once');

   calc_model_prior_test;



function calc_model_prior_test;
% Verify model prior calcs
% Imported from: calc_model_prior_test.m 3127 2012-06-08 16:19:25Z bgrychtol

% TODO: also test the various inverse prior calls

   imdl= mk_common_model('c2c2',16);
   try; imdl= rmfield(imdl,'RtR_prior'); end
   try; imdl= rmfield(imdl,'R_prior');   end

   any_priors= {@prior_tikhonov, ...
                @prior_noser, ...
                @prior_gaussian_HPF, ...
                @prior_laplace};

  test_R=[ 0.041630544712308
           6.773110974081427e-04
           0.036206173920553
           2.592951965329636
           0.002008656225401];

   R_priors=   {any_priors{:}, ...
                @prior_TV};

   % Call R_priors as R_priors
   eidors_cache clear
   for i = 1:length(R_priors); p = R_priors{i};
      inv_mdl= imdl;
      inv_mdl.R_prior= p;
      R= calc_R_prior(inv_mdl);
%     test = std(R'*R,0,'all'); 
      test = std(reshape(R'*R,[],1),0); % compatible with R2017b
      unit_test_cmp(['R_prior (R''*R): ', func2str(p)], test, test_R(i), 1e-12)
   end
      
   % Call R_priors as RtR_priors
   eidors_cache clear
   for i = 1:length(R_priors); p = R_priors{i};
      inv_mdl= imdl;
      inv_mdl.R_prior= p;
      RtR= calc_RtR_prior(inv_mdl);
      test = std(RtR(:));
      unit_test_cmp(['R_prior (R''*R): ', func2str(p)], test, test_R(i), 1e-12)
   end

  test_RtR=[ 0.041630544712308
             0.003569905276849
             0.037777196349047
             0.282597508516543
             0.006965047087020];
      
   % Call RtR_priors as RtR_priors
   eidors_cache clear
   for i = 1:length(R_priors); p = R_priors{i};
      inv_mdl= imdl;
      inv_mdl.RtR_prior= p;
      RtR= calc_RtR_prior(inv_mdl);
      if diff(size(RtR))~=0  % non-square
         fprintf('RtR_prior: %20s  RtR_condest= NON-SQUARE\n', func2str(p) );
      end
      test = std(RtR(:));
      unit_test_cmp(['RtR_prior (RtR): ', func2str(p)], test, test_RtR(i), 1e-12)
   end

   test_R_= [ 0.041630544712308
              0.010392960833110
              0.040302448507613
              0.100634712398168];
   % Call RtR_priors as R_priors
   eidors_cache clear
   for i = 1:length(R_priors); p = R_priors{i};
      inv_mdl= imdl;
      inv_mdl.RtR_prior= p;
      if strcmp(func2str(p), 'prior_TV')
         continue; % not fair to ask it to ichol a non-square matrix
      end
      R= calc_R_prior(inv_mdl);
      test = std(R(:));
%     fprintf('RtR_prior: %20s  R_condest= %5.4g\n', func2str(p), condest(R'*R));
      if exist('OCTAVE_VERSION') && strcmp(func2str(p),'prior_gaussian_HPF')
         eidors_msg('prior_gaussian_HPF test not valid in octave (need ldl)');
      else
         unit_test_cmp(['RtR_prior (R): ', func2str(p)], test, test_R_(i), 1e-5)
      end
   end
