function J = calc_jacobian( fwd_model, img)
% CALC_JACOBIAN: calculate jacobian from an inv_model
% 
%  J = calc_jacobian( img )
%      calc Jacobian on img.fwd_model at conductivity given
%      in image (fwd_model is for forward and reconstruction)
% 
% The actual work is done by the jacobian calculator specified in 
%    img.fwd_model.jacobian, unless that field is numeric, in which case
%    calc_jacobian returns its contents.
%
% For reconstructions on dual meshes, the interpolation matrix
%    is defined as img.fwd_model.coarse2fine. This takes
%    coarse2fine * x_coarse = x_fine
%
% If the underlying jacobian calculator doesn't understand dual
%    meshes, then calc_jacobian will automatically postmultiply
%    by fwd_model.coarse2fine.
%
% img       is an image structure, with 'elem_data' or
%           'node_data' parameters

% (C) 2005-08 Andy Adler. License: GPL version 2 or version 3
% $Id: calc_jacobian.m 6716 2024-04-02 17:59:16Z aadler $

if ischar(fwd_model) && strcmp(fwd_model,'UNIT_TEST'); do_unit_test; return; end

if nargin == 1
   img= fwd_model;
else
   warning('EIDORS:DeprecatedInterface', ...
      ['Calling CALC_JACOBIAN with two arguments is deprecated and will cause' ...
       ' an error in a future version. First argument ignored.']);
end
ws = warning('query','EIDORS:DeprecatedInterface');
warning off EIDORS:DeprecatedInterface

try 
   fwd_model= img.fwd_model;
catch
   error('CALC_JACOBIAN requires an eidors image structure');
end

fwd_model_check(fwd_model);

if isnumeric(fwd_model.jacobian)             % we have the Jacobian matrix
   J = fwd_model.jacobian;
else                                         % we need to calculate
   
   copt.cache_obj= jacobian_cache_params( fwd_model, img );
   copt.fstr = 'jacobian';
   try
       fwd_model.jacobian = str2func(fwd_model.jacobian);
   end
   J = eidors_cache(fwd_model.jacobian, {fwd_model, img}, copt);
   
end

if isfield(fwd_model,'coarse2fine')
   c2f= fwd_model.coarse2fine;
   if size(J,2)==size(c2f,1)
%     calc_jacobian did not take into account the coarse2fine
      J=J*c2f;
   end
end

warning(ws.state, 'EIDORS:DeprecatedInterface');

        



% Make the Jacobian only depend on 
function cache_obj= jacobian_cache_params( fwd_model, img );
   img = data_mapper(img);
   if isfield(img, 'elem_data')
      cache_obj = {fwd_model, img.elem_data, img.current_params};
   elseif isfield(img, 'node_data')
      cache_obj = {fwd_model, img.node_data, img.current_params};
   else
      error('calc_jacobian: execting elem_data or node_data in image');
   end

function fwd_model_check(fmdl)
   try; if fmdl.jacobian_dont_check
      return
   end; end
   pp = fwd_model_parameters(fmdl); % they cache, so no problem
   if pp.n_elec == 0
       error('Cannot calculate Jacobian. No electrodes found.');
   end
   if pp.n_stim == 0
       error('Cannot calculate Jacobian. No stimulation found.');
   end
   if pp.n_meas == 0
       error('Cannot calculate Jacobian. No measurements found.');
   end

function do_unit_test
    delta = 1e-6;
    testvec= [5,20,40,130];
    mdl= test_aa_mdl2;
    run_jacobian_test( mdl, delta, testvec );
    mdl = mdl_normalize(mdl,1); mdl.name = [mdl.name,':normalize'];
    run_jacobian_test( mdl, delta, testvec );

    mdl= test_aa_mdl3;
    run_jacobian_test( mdl, delta, testvec );
    mdl = mdl_normalize(mdl,1); mdl.name = [mdl.name,':normalize'];
    run_jacobian_test( mdl, delta, testvec );

    warning('off','EIDORS:deprecated');
    mdl= test_np_mdl;
    % Need lower tol because np models use a solver with lower tol
    run_jacobian_test( mdl, delta, testvec, 1e-3 );
    mdl = mdl_normalize(mdl,1); mdl.name = [mdl.name,':normalize'];
    run_jacobian_test( mdl, delta, testvec, 1e-3 );

function run_jacobian_test( mdl, delta, testvec,tol ); 
    img= mk_image(mdl,1);
    homg_data=fwd_solve( img);

    J= calc_jacobian( img );
    if nargin<4;tol= 1e-5;end
    tol = tol*std(J(:));

    % J = dF/dx = [F(x+d)  - F(x)]/d
    sumdiff= 0;
    bkgnd_elem_data= img.elem_data;
    for testelem = testvec
       img.elem_data= bkgnd_elem_data;
       img.elem_data(testelem)= bkgnd_elem_data(testelem)+delta;
       inh_data=fwd_solve( img);

       if mdl_normalize(mdl)
          simJ= 1/delta* (inh_data.meas ./ homg_data.meas - 1);
       else
          simJ= 1/delta* (inh_data.meas-homg_data.meas);
       end
       
       plot([J(:,testelem) simJ]);
       sumdiff = sumdiff + std( J(:,testelem) - simJ );
    end

    dif= sumdiff/length(testvec);
    unit_test_cmp(['Jacobian calc: ',mdl.name],sumdiff/length(testvec),0,tol);

function mdl= test_aa_mdl2;
    n_elec= 16;
    n_rings= 1;
    options = {'no_meas_current','no_rotate_meas'};
    params= mk_circ_tank(8, [], n_elec);

    params.stimulation= mk_stim_patterns(n_elec, n_rings, '{ad}','{ad}', ...
                                options, 10);
    params.solve=      'fwd_solve_1st_order';
    params.system_mat= 'system_mat_1st_order';
    params.jacobian=   'jacobian_adjoint';
    params.normalize_measurements = 0;
    mdl = eidors_obj('fwd_model', params);
    mdl.name= 'AA_1996 mdl';



function mdl= test_aa_mdl3;
    i_mdl = mk_common_model('b3cz2',16);
    mdl= i_mdl.fwd_model;
    mdl.name= 'AA_1996 mdl';
    mdl.solve=      'fwd_solve_1st_order';
    mdl.system_mat= 'system_mat_1st_order';
    mdl.jacobian=   'jacobian_adjoint';
    
    

function mdl= test_np_mdl;
    i_mdl = mk_common_model('n3r2',[16,2]);
    mdl= i_mdl.fwd_model;
    mdl.name=     'NP_2003 mdl';
    mdl.solve=      'np_fwd_solve';
    mdl.system_mat= 'np_calc_system_mat';
    mdl.jacobian=   'np_calc_jacobian';

function mdl= test_ms_mdl;
    i_mdl = mk_common_model('n3r2',16);
    mdl= i_mdl.fwd_model;
    mdl.name=       'MS_2005 mdl';
    mdl.solve=      'np_fwd_solve';
    mdl.system_mat= 'np_calc_system_mat';
    mdl.jacobian=   'ms_calc_jacobian';
