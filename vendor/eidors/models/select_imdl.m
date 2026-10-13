function [inv_mdl,opt_out]= select_imdl( mdl, options )
% SELECT_IMDL: select pre-packaged inverse model features
% inv_mdl= select_imdl( mdl, options )
%
%   mdl - inv_model structure - parameters are replaced as specified
%    OR
%   mdl - fwd_model OR image
%
% OPTIONS => {'opt1','opt2'} options are processed in the order specified
% if OPTIONS is char, treated as first option
%
% Available options are:
%
% 'Basic GN dif';   Basic GN one step difference solver with Laplace prior
% 'Basic GN abs';   Basic Gauss-Newton absolute solver with Laplace prior
% 'NOSER dif';      Basic GN one step difference solver with NOSER prior 
% 'Nodal GN dif';   Basic GN solver, solves onto nodes
% 'TV solve dif':   Total Variation PDIPM difference solver 
% 'GREIT':          Algorithm of mk_GREIT_model
%   parameters provided as select_imdl(fmdl,{'GREIT:NF=0.3 64x64 rad=0.25'});
%   defaults NF=0.5 32x32 rad=0.2
% 'GREIT':          3D GREIT if parameters:
%     {'GREIT:NF=0.3 32x32x6'});   % default vrange=.2
%     {'GREIT:NF=0.3 vrange=0.3 32x32x6'});
%   Here 6 slices cover +/-vrange*max(diam) from verticle centre
%
% 'Elec Move GN':   One step GN difference solver with compensation for
%                   electrode movement. Conductivity prior, hyperparameter
%                   and Jacobian will be preserved if specified in the
%                   model
% 'Choose NF=1.0';  Choose hyperparameter value appropriate for specified noise figure (NF)

% (C) 2010-2024 Andy Adler. License: GPL version 2 or version 3
% $Id: select_imdl.m 6972 2024-10-03 11:40:00Z aadler $

if ischar(mdl) && strcmp(mdl,'UNIT_TEST'); do_unit_test; return; end

if nargin == 1; options = {}; end

switch mdl.type
  case 'inv_model'; inv_mdl = mdl;
  case 'fwd_model'; inv_mdl = basic_imdl( mdl );
  case 'image';     inv_mdl = basic_imdl( mdl.fwd_model, mdl.elem_data );
  otherwise;        error('select_imdl: expects inv_model or fwd_model input');
end

opt_out = struct(); % default setting
if ischar(options); options = {options}; end
for i=1:length(options);
% Pre matlab v7 code
% [s,f,tok]= regexp(options{i},'(.[^=:]*)[=:]?(.*)');
% tok = tok{1};
% for t=1:size(tok,1);
%    opt{t} = options{i}(tok(t,1):tok(t,2));
% end
  % split the option on an equals sign
  opt= regexp(options{i},'(.[^=:]*)[=:]?(.*)','tokens'); opt= opt{1};
  switch opt{1}
    case 'NOSER dif';       inv_mdl = NOSER_dif( inv_mdl );
    case 'Basic GN dif';    inv_mdl = Basic_GN_Dif( inv_mdl );
    case 'Basic GN abs';    inv_mdl = Basic_GN_Abs( inv_mdl );
    case 'Nodal GN dif';    inv_mdl = Nodal_GN_Dif( inv_mdl );
    case 'TV solve dif';    inv_mdl = TV_solve_Dif( inv_mdl );
    case 'Elec Move GN';    inv_mdl = Elec_Move_GN( inv_mdl );
    case 'Choose NF';       inv_mdl = Choose_NF( inv_mdl, str2num(opt{2}) );
    case 'GREIT';           [inv_mdl,opt_out] = GREIT(inv_mdl, opt{2});
    
    otherwise; error('option {%s} not understood', options{i});
  end
end

% standard field order
inv_mdl = eidors_obj('inv_model',inv_mdl);

% check we are giving back a good specimen
valid_inv_model(inv_mdl);


function imdl = basic_imdl( fmdl, elem_data );
   if nargin<=1; elem_data = 1; end
   imdl.name= 'Basic imdl from select_imdl';
   imdl.type= 'inv_model';

   imdl.solve= 'eidors_default';
   imdl.hyperparameter.value = .01;
   imdl.RtR_prior = 'eidors_default';
   imdl.jacobian_bkgnd.value = elem_data;
   imdl.reconst_type= 'difference';
   imdl.fwd_model = fmdl;

function imdl = NOSER_dif( imdl );
   imdl.RtR_prior = @prior_noser;
   try; imdl = rmfield(imdl,'R_prior'); end
   imdl.hyperparameter.value = .03;
   imdl.solve= @inv_solve_diff_GN_one_step;
   imdl.reconst_type= 'difference';

function imdl = Basic_GN_Dif( imdl );
   imdl.RtR_prior = @prior_laplace;
   try; imdl = rmfield(imdl,'R_prior'); end
   imdl.solve= @inv_solve_diff_GN_one_step;
   imdl.reconst_type= 'difference';

function imdl = Basic_GN_Abs( imdl );
   imdl.RtR_prior = @prior_laplace;
   try; imdl = rmfield(imdl,'R_prior'); end
   imdl.solve= @inv_solve_gn;
   imdl.inv_solve_gn.max_iterations= 10;
   imdl.reconst_type= 'absolute';

function imdl = TV_solve_Dif( imdl );
   imdl.R_prior = @prior_TV;
   try; imdl = rmfield(imdl,'RtR_prior'); end
   imdl.solve= @inv_solve_TV_pdipm;
   imdl.reconst_type= 'difference';
   imdl.inv_solve.max_iterations= 15;
   imdl.hyperparameter.value = 1e-1;
   imdl.inv_solve.term_tolerance = 1e-3;

function imdl = Elec_Move_GN( imdl );
   % keep previous model as conductivity jacobian, so it should be ok
   imdl.fwd_model.conductivity_jacobian = imdl.fwd_model.jacobian; 
   imdl.fwd_model.jacobian = @jacobian_movement;
   
   % keep previous prior
   try
       imdl.prior_movement.RegC.func = imdl.RtR_prior;
   catch
       imdl.prior_movement.RegC.func = @prior_laplace;
       %  imdl.prior_movement.RegC.func = @prior_gaussian_HPF; % not OK for 3D
   end
   
   imdl.RtR_prior =          @prior_movement;
   MV_prior = 1./mean( std( imdl.fwd_model.nodes ));
   imdl.prior_movement.parameters = MV_prior;
   
   imdl.solve= @inv_solve_diff_GN_one_step;
   
   % keep hyperparameter
   try 
       hp = imdl.hyperparameter.value;
   catch
       hp = 0.03;
   end
   imdl.hyperparameter.value = hp;
   
   try
      n_elems = size(imdl.rec_model.coarse2fine,2);
   catch
      n_elems = size(imdl.fwd_model.elems,1);
   end
   imdl.inv_solve.select_parameters = 1:n_elems;

   imdl.prior_use_fwd_not_rec = 1; % for c2f mapping

function imdl = Nodal_GN_Dif( imdl );
   imdl.solve = @nodal_solve;

function [imdl,opt] = GREIT(inv_mdl, opts)
   opt.square_pixels = true;
   opt.noise_figure = 0.5;
   opt.imgsz = [32 32];
   % Allow error compensation afterward
   opt.keep_model_components = true;
   radius = 0.2;

   vrange = 0.2; % TODO: add adjust
   matches = regexp(opts,'\S+','match');
   for i=1:length(matches); mi = matches{i};
      tok= regexp(mi,'NF=([\d\.]+)','tokens');
      if ~isempty(tok);
         opt.noise_figure = str2num(tok{1}{1});
      end

      tok= regexp(mi,'rad=([\d\.]+)','tokens');
      if ~isempty(tok);
         radius = str2num(tok{1}{1});
      end

      tok= regexp(mi,'vrange=([\d\.]+)','tokens');
      if ~isempty(tok);
         vrange = str2num(tok{1}{1});
      end

      tok= regexp(mi,'(\d+)x(\d+)','tokens');
      if ~isempty(tok);
         opt.imgsz = [str2num(tok{1}{1}), str2num(tok{1}{2})];
      end

      % 3D GREIT
      tok= regexp(mi,'(\d+)x(\d+)x(\d+)','tokens');
      if ~isempty(tok);
         tokA=str2num(tok{1}{1});
         tokB=str2num(tok{1}{2});
         tokC=str2num(tok{1}{3});
         [minf,maxf] = bounds(inv_mdl.fwd_model.nodes);
         vctr = mean([minf(3),maxf(3)]);
         mrad = max( maxf(1:2) - minf(1:2) );

         vopt.imgsz = [tokA, tokB];
         vopt.square_pixels = true;
         vopt.zvec = linspace(-1,1,tokC+1)*mrad*vrange + vctr;
         vopt.save_memory = 2;

% GREIT 3D with a 1x32 electrode layout
         [inv_mdl,opt.distr] = GREIT3D_distribution(inv_mdl, vopt);
      end
   end

   imdl= mk_GREIT_model(inv_mdl, radius, [], opt);
   

function imdl = Choose_NF( imdl, NF_req );
   if ~strcmp(imdl.reconst_type, 'difference');
      error('Choose NF only works for difference solvers right now');
   end

% Find 4 elems in mesh ctr to be the NF target elems
   xyz_elems = interp_mesh( imdl.fwd_model );
   ctr_elems = mean(xyz_elems, 1);
   xyz_elems = xyz_elems - ones(size(xyz_elems,1),1)*ctr_elems;
   d_elems   = sqrt( sum( xyz_elems.^2, 2 ));
   [jnk, e_idx] = sort(d_elems);

   imdl.hyperparameter.tgt_elems = e_idx(1:4);
   imdl.hyperparameter.noise_figure = NF_req;

   sv_log = eidors_msg('log_level'); eidors_msg('log_level', 1);
   HP = choose_noise_figure( imdl );
   eidors_msg('log_level', sv_log);

   imdl.hyperparameter.value = HP;


function do_unit_test
% Test difference solvers on lung images
   load montreal_data_1995;
   imdl = mk_common_model('b2t3',16); 

   i=0;while true; i=i+1; % Break when finished
      vh = zc_resp(:,1); vi= zc_resp(:,23);
      fprintf('solver #%d\n',i);
      switch i
         case 01;
            imdl0 = select_imdl( imdl );
         case 02;
            imdl0 = select_imdl( imdl.fwd_model );
         case 03;
            imdl0 = select_imdl( imdl, {'NOSER dif'} );
         case 04;
            imdl0 = select_imdl( imdl, {'NOSER dif','Choose NF=1.1'});
         case 05;
            imdl0 = select_imdl( imdl, {'Basic GN dif'} );
         case 06;
            imdl0 = select_imdl( imdl, {'TV solve dif'} );
         case 07;
            imdl0 = select_imdl( imdl.fwd_model, {'Basic GN dif','Elec Move GN'} );
         case 08;
            imdl0 = select_imdl( imdl.fwd_model, {'Basic GN dif','Elec Move GN','Choose NF=0.5'} );
         case 09;
            imdl0 = select_imdl( imdl, {'Elec Move GN'} );
         case 10;
            imdl0 = mk_common_model('b2C2',16); 
            imdl0 = select_imdl( imdl0, {'Elec Move GN'} );
         case 11;
            imdl0 = select_imdl( imdl, {'Nodal GN dif'} );
         case 12;
            imdl0 = select_imdl( imdl, {'Nodal GN dif', 'Choose NF=0.50'} );
         case 13;
            fmdl = mk_library_model('adult_male_16el');
            [stim,msel] = mk_stim_patterns(16,1,[0,1],[0,1],{},1);
            fmdl.stimulation = stim; 
            fmdl.meas_select = msel; 
            imdl0 = select_imdl( fmdl, 'GREIT');

         case 14;
            imdl0 = mk_common_model('b2C2',16); 
            imdl0 = select_imdl( imdl0, {'Basic GN dif', 'TV solve dif'} );
            imdl0.inv_solve.max_iterations= 2;
            imdl0 = select_imdl( imdl0, {'Choose NF=0.8'} );
            [vh,vi] = simulate_movement(mk_image(imdl0), [0;0.5;0.35]);
         case 15;
            imdl0 = mk_common_model('b2C2',16); 
            imdl0 = select_imdl( imdl0, {'Nodal GN dif', 'Choose NF=0.50'} );
            [vh,vi] = simulate_movement(mk_image(imdl0), [0;0.5;0.05]);
         case 16;
            imdl0 = mk_common_model('b2C2',16); 
            imdl0 = select_imdl( imdl0, {'Basic GN abs'} );
            [vh,vi] = simulate_movement(mk_image(imdl0), [0;0.5;0.05]);
         case 17;
            imdl = mk_common_model('n3r2',[16,2]); fmdl = imdl.fwd_model;
            fmdl.nodes(:,3) = fmdl.nodes(:,3) - 1.5;
            imdl0= select_imdl(fmdl,{'GREIT:NF=0.3 64x64 rad=0.25'});
            img = mk_image(fmdl,1); vh = fwd_solve(img);
            img.elem_data(fmdl.demo_targ_elems.A) = 1.2;
            img.elem_data(fmdl.demo_targ_elems.B) = 0.8;
                                    vi = fwd_solve(img);
           
         case 18;
            imdl = mk_common_model('n3r2',[16,2]); fmdl = imdl.fwd_model;
            fmdl.nodes(:,3) = fmdl.nodes(:,3) - 1.5;
            imdl0= select_imdl(fmdl,{'GREIT:NF=0.3 32x32 rad=0.25'});
            img = mk_image(fmdl,1); vh = fwd_solve(img);
            img.elem_data(fmdl.demo_targ_elems.A) = 1.2;
            img.elem_data(fmdl.demo_targ_elems.B) = 0.8;
                                    vi = fwd_solve(img);
            
         otherwise; break
      end;

%     disp([i,imdl0.hyperparameter.value]);
      if strcmp( imdl0.reconst_type, 'absolute')
         imgr = inv_solve( imdl0, vi);
      else
         imgr = inv_solve( imdl0, vh, vi);
      end
      subplot(4,5,i); show_slices( imgr, [inf,inf,0] );
   end
