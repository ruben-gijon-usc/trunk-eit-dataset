function imdl= GREIT_errors(imdli, opts, data )
% GREIT_errors: Add Error Compensation to GREIT-type model
% imdl= imdl_errors(imdli, options )
%  imdl - inv_model structure
%
% OPTIONS => {'opt1','opt2'} options are processed in the order specified
% if OPTIONS is char, treated as first option
%
% Available options are:
%
% 'Channel:[list]' or 'Channel:[list] SNR=100'  # remove individual channels
% 'Electrode:[list]' or 'Electrode:[list] SNR=100' # remove electrodes
% 'RecipError:' or 'RecipError: Tau=1e-3'       # use calc_reciproc_error
%
% Data is a sequence of measured data to model electrode errors with
%
% See also: select_imdl
%
% Model must have a imdl.solve_use_matrix field with RM, M, PJt fields
%  these are created with keep_model_components option to mk_GREIT_model

% (C) 2024 Andy Adler. License: GPL version 2 or version 3
% $Id: GREIT_errors.m 6682 2024-03-14 19:46:10Z aadler $

if ischar(imdli) && strcmp(imdli,'UNIT_TEST'); do_unit_test; return; end
if nargin<3; data=[]; end

imdl = imdli;
if ischar(opts); opts = {opts}; end
for i=1:length(opts);
   if iscell(opts); opti = opts{i};
   else           ; opti = opts; end
   imdl = process(imdl, opti, data);
end

function imdl = process(imdli, opti, data);
   if ~ischar(opti); error('Expecting string options'); end
   SNR = 1000; % Default values
   Tau = -3e-4;
   imdl = imdli;
   PJt        = imdli.solve_use_matrix.PJt;
   M          = imdli.solve_use_matrix.M;
   noiselev   = imdli.solve_use_matrix.noiselev;

   % split the option on an equals sign
   opt= regexp(opti,'(.[^:]*):?(.\S*)\s*(.*)','tokens'); opt= opt{1};
   matches = regexp(opt{3},'(\S+)=(\S+)','tokens');
   for i=1:length(matches); switch matches{i}{1}
      case 'SNR'; SNR = str2num(matches{i}{2});
      case 'Tau'; Tau =-abs(str2num(matches{i}{2})) 
      otherwise;  error('unknown parameter');
   end; end

   switch lower(opt{1});
      case 'channel';
         noise_covar = sparse(size(M,1),size(M,2));
         for ch= str2num(opt{2});
            noise_covar(ch,ch) = SNR;
         end
      case 'electrode';
         imdli.meas_icov_rm_elecs.elec_list = str2num(opt{2});
         imdli.meas_icov_rm_elecs.exponent  = -1;
         imdli.meas_icov_rm_elecs.SNR       = SNR;
         noise_covar = meas_icov_rm_elecs(imdli);
      case 'reciperror';
         imdl.calc_reciproc_error.tau = Tau;
         noise_covar= calc_reciproc_error( imdl, data );
      otherwise error('option not understood');
   end

   M   = M + noiselev^2 * noise_covar;
   imdl.solve_use_matrix.RM = left_divide(M.',PJt.').';    %PJt/M;
   

function do_unit_test
   for sp=5:9; switch sp
      case 1; [vv,imdl] = test_data(1);
      case 2; [vv,imdl] = test_data(2);
      case 3; [vv,imdl] = test_data(2);
         imdl = GREIT_errors(imdl,'Electrode:[2]');
      case 4; [vv,imdl] = test_data(2);
         imdl = GREIT_errors(imdl,'Channel:[3]');
      case 5; [vv,imdl] = test_data(2);
         imdl = GREIT_errors(imdl,'Channel:[3] SNR=10000');
      case 6; [vv,imdl] = test_data(2);
         imdl = GREIT_errors(imdl,'RecipError: Tau=.0001',vv);
      case 7; [vv,imdl] = test_data(2);
         imdl = GREIT_errors(imdl,'RecipError:',vv);
      otherwise; break;
  
      end
      imgr = inv_solve(imdl,vv(:,1),vv(:,end));
      imgr.calc_colours.ref_level = 0;
      subplot(3,3,sp); hh=show_fem(imgr);
      set(hh,'EdgeColor','none');
   end

function [vv,imdl] = test_data(mode);
   fmdl = mk_library_model('adult_male_16el_lungs');
   fmdl.stimulation = mk_stim_patterns(16,1,[0,1],[0,1],{});
   img = mk_image(fmdl,1);
   k=0;for sig = linspace(0.2,0.1,50); k=k+1;
      img.elem_data(vertcat(fmdl.mat_idx{2:3}))=sig;
      vv(:,k) = getfield(fwd_solve(img),'meas');
   end
   imdl = select_imdl(fmdl,{'GREIT:NF=0.3 64x64 rad=0.25'});

   rng('default');
   switch mode
      case 1;
      case 2; vv(3,:) = 0.1*randn(size( vv(3,:) ));
      otherwise; error 'huh?'
   end
