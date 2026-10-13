function img = inv_solve_d_bar(inv_model, data1, data2)
%D-Bar reconstruction
%inv_model is an inverse model
% data1 is the inhomogene data
% data2 is backgounrd of 1 S/m

% (C) 2023 Joeran Rixen. Licensed under GPL version 2 or 3

%
% CITATION_REQUEST:
% AUTHOR: Rixen et al.
% TITLE: The D-Bar Algorithm Fusing Electrical Impedance Tomography
% with A Priori Radar Data: A Hands-On Analysis
% JOURNAL: Algorithms
% VOL: 16
% NUM: 1
% YEAR: 2023
% LINK: https://doi.org/10.3390/a16010043
% DOI: 10.3390/a16010043

% TODO:
% - remove elec_area hardcode
% - tests for different stimulation patterns
% - see how to remove stim_pattern_mat


    if ischar(inv_model) && strcmp(inv_model,'UNIT_TEST'); test_d_bar; return; end

    citeme(mfilename);

    fmdl = inv_model.fwd_model;
    n_elec = num_elecs(fmdl);
    
    try
        trunc_rad = inv_model.hyperparameter.truncation_radius;
        grid_size = inv_model.hyperparameter.grid_size;
    catch
        error('This does not look like a valid d-bar model!')
    end

    %% Calculations regarding the Dirichlet to Neumann map
    stim_pattern_mat = create_stim_pattern_matrix(fmdl);
    stim_pattern_mat = stim_pattern_mat./sqrt(sum(stim_pattern_mat.^2,1));

    %% Calculations regarding the scattering transformation
    elec_pos_angle = get_elec_pos_angle(fmdl);
    
    %compute the grid size
    M = 2^grid_size;
    %step size between each point in K
    stepsize_grid = 2*trunc_rad/(M - 1);
    grid_base_vec = [-trunc_rad:stepsize_grid:trunc_rad];
    [K_x, K_y] = meshgrid(grid_base_vec);
    %creation of K
    K = K_x + 1i*K_y;
    %boolean mask of which points are actually inside the truncation radius
    K_inside_R = K( abs(K) < trunc_rad);
    
    del_L = dirichlet_neumann_difference(fmdl,data1,data2);
    
    %computation of the scattering transform
    t_aporx = zeros(length(K_inside_R),1);
    for i = 1:length(K_inside_R)
        k = K_inside_R(i);
        c_k = (full(stim_pattern_mat))'*exp(1i*k*exp(1i*elec_pos_angle));%%electrode_pos
        d_k = (full(stim_pattern_mat))'*exp(1i*conj(k)*conj(exp(1i*elec_pos_angle)));
    
        t_aporx(i) = conj(d_k)'*(del_L*c_k);
    end
    
    %% Setting up of the calculations for solving the D-Bar equation
    %calculation of the slightly larger Grid Q (comapred to K) for the Greens function
    begin_Q = trunc_rad*((2*M - 1)/(M - 1));
    grid_vec_Q = [-begin_Q:2*trunc_rad/(M-1):begin_Q];
    [Q_x, Q_y] = meshgrid(grid_vec_Q);
    Q = Q_x + 1i*Q_y;
    
    % factor that decides how big the smoothing zone outside of R is
    smooth_zone_factor = 1/4;
    % calcualte G bar
    G_bar = 1./(pi*Q);
    % set everything to 0 outisde the smoothing zone
    G_bar(abs(Q) > trunc_rad*(2 + smooth_zone_factor)) = 0;
    % get index of the smoothing zone
    smooth_zone = abs(Q) > trunc_rad*(2) & abs(Q) < trunc_rad*(2 + smooth_zone_factor);
    % apply the smoothing (in this case linear) to G_bar
    G_bar(smooth_zone) = G_bar(smooth_zone).*((trunc_rad*(2 + smooth_zone_factor) - abs(Q(smooth_zone)))/(trunc_rad*smooth_zone_factor));
    
    % make the scattering transform compatible to the larger grid Q
    scat = zeros(size(K));
    rad_idx = abs(K) < trunc_rad;
    scat(rad_idx) = t_aporx;
    scat = padarray_EIDORS(scat, [M/2 M/2], 0, 'both');
    
    % use the FFT trick by Mueller and Siltanen
    %(due to Matlabs indexing fftshift is needed before apllying fft2)
    G_bar_fft = fft2(fftshift(G_bar));
    
    % calcualte the first part of T (this is independnt of the location z)
    T_R_first = scat./(4*pi*conj(Q));
    
    
    % Stuff needed for the computation, this is not directly related to the D-Bar algorithm
    % index of which vlaues are inside R
    rad_idx_dbar = abs(Q) < trunc_rad;
    % number of points inside R
    num_pts_in_R = sum(rad_idx_dbar(:));
    
    % Construct right hand side of the Dbar equation as the initial guess
    right_side_eq = [ones(num_pts_in_R,1);zeros(num_pts_in_R,1)];
    
    recon_cords = interp_mesh(fmdl);
    % transform the reconstruction points into complex coordiantes
    recon_pts = recon_cords(:, 1) + 1i*recon_cords(:, 2);
    num_rec_pts  = length(recon_pts);
    
    mu_final = ones(num_rec_pts,1);
    iniguess = [ones(num_pts_in_R,1);zeros(num_pts_in_R,1)];
    
    
    middle_val = size(T_R_first, 1)/2;
    % When you have the parralel computation toolkit feel free to use parfor instead of for
    for iii = 1:num_rec_pts
        % get positions for recosntruction
        z = recon_pts(iii);
    
        % multiplicator for the D-Bar function
        T_R = T_R_first.*(exp(-1i*(Q*z+conj(Q*z))));
    
        % Solve D-Bar equation; Note that instead of a matrix a function is
        % given, the input aargument for that function follow after the
        % second []
        [sol, ~] = gmres(@DB_oper_demyst, right_side_eq, 40, 1e-4, 450, [], [], iniguess, M*2, rad_idx_dbar, num_pts_in_R, 2*trunc_rad/(M - 1), G_bar_fft, T_R);
    
        %use sol to get mu for the recosntruction
        mu = zeros(size(Q));
        mu(rad_idx_dbar) = sol(1:num_pts_in_R) + 1i*sol((num_pts_in_R+1):end);
    
        % get mu from interpolation of the center
        mu_final(iii) = mean([mu(middle_val,middle_val), mu(middle_val+1,middle_val), mu(middle_val, middle_val+1), mu(middle_val+1, middle_val+1)]);
    
    end
    img = struct;
    img.name= ['solved by D-Bar'];
    img.elem_data = get_conductivity(mu_final);
    img.fwd_model= fmdl;
end


function stim_pattern_mat = create_stim_pattern_matrix(fmdl)
    % converts the stimpattern in the model into matrix form for ease of
    % computation
    n_elec = length(fmdl.electrode);
    stim_pattern_mat = zeros(n_elec, n_elec-1);
    
    for i = 1:n_elec-1
        stim_pattern_mat(:, i) = full(fmdl.stimulation(i).stim_pattern);
    end
end

function elec_pos_angle = get_elec_pos_angle(fmdl)
    % calculates the position of the electrodes with respect to it's angle
    n_elec = length(fmdl.electrode);
    elec_pos_cart = zeros(n_elec, 2);
    
    for i = 1:n_elec
        elec_pos_cart(i, :) = [mean(fmdl.nodes(fmdl.electrode(i).nodes, 1)), mean(fmdl.nodes(fmdl.electrode(i).nodes, 2))];
    end
    
    elec_pos_cart = [elec_pos_cart(:, 1) - mean(elec_pos_cart(:, 1)), elec_pos_cart(:, 2) - mean(elec_pos_cart(:, 2))];
    
    elec_pos_angle = atan2(elec_pos_cart(:, 2), elec_pos_cart(:, 1));
    
    elec_pos_angle(elec_pos_angle < 0) = elec_pos_angle(elec_pos_angle < 0) + 2*pi;
end

function B = padarray_EIDORS(A,padsize,padval,direction)
% padarray function is not availabe in standard matlan.
    if nargin < 4
        direction = 'both';
    end
    if nargin < 3
        padval = 0;
    end
    if strcmp(direction,'both')
        B = padval*ones(size(A)+2*padsize);
        B(padsize(1)+1:end-padsize(1),padsize(2)+1:end-padsize(2)) = A;
    elseif strcmp(direction,'pre')
        B = padval*ones(size(A)+padsize);
        B(padsize(1)+1:end,padsize(2)+1:end) = A;
    elseif strcmp(direction,'post')
        B = padval*ones(size(A)+padsize);
        B(1:size(A,1),1:size(A,2)) = A;
    end
end

function [conductivity] = get_conductivity(mu)
    %computes the conductivty from the solution of the D-Bar equation:
    
    %mu <number of points to reconstruct>:
    %solution of the D-Bar equation
    
    conductivity = real(mu.^2);
end

function [fmdl, inv_model] = mk_dbar_model(num_electrs)
    % creates a forward model compatible to the d-bar algorithm
    % and creates a backwards model for the d-bar algorithm
    fmdl = ng_mk_cyl_models(.5,[num_electrs,0.25],[0.075]);
    [stim, ~]= mk_stim_patterns(num_electrs, 1, '{trig}', '{mono}', {'meas_current'}, 0.1);
    fmdl.stimulation = stim;
    
    fmdl_2d = ng_mk_cyl_models([0,1,0.05],[num_electrs,0],[0.075]);
    fmdl_2d.stimulation = stim;

    inv_model = struct;
    inv_model.name = 'EIDORS D-Bar model';
    inv_model.solve = 'inv_solve_d_bar';
    inv_model.RtR_prior = 'eidors_default';

    % set multiple hyperparameters
    inv_model.hyperparameter.truncation_radius = 3.0;
    inv_model.hyperparameter.grid_size = 5;
   
    inv_model.jacobian_bkgnd = 1;
    inv_model.type = 'inv_model';
    inv_model.fwd_model = fmdl_2d;
end

function result = DB_oper_demyst(sol, M, rad_idx_dbar, num_pts_in_R, h, G_bar_fft, TR)
    %rearranges the solution of the D-Bar algorithm to fit the square, perform
    %the D-Bar operation and transform it back:

    %sol <(total number of gridpoints)*2>: solution of the D-Bar equation so
    %far
    
    %M <(size of Q)*2>: size of Q
    
    %rad_idx_dbar <size of Q>*<size of Q>: indices whether grid points are
    %inside R or not
    
    %num_pts_in_R <1>: number of points inside R
    
    %h <1>: spacing of the grid
    
    %G_bar_fft <size of Q>*<size of Q>:  FFT trick operator
    
    
    sol_square = zeros(M, M);
    sol_square(rad_idx_dbar) = sol(1:num_pts_in_R) + 1i*sol((num_pts_in_R+1):end);
    
    % Apply real-linear operator %%eqn. 15.65 from "Linear- and non_liner
    % inverse problems"
    result = sol_square - h^2*ifft2(G_bar_fft .* fft2( TR.*conj(sol_square) ));
    
    % Construct result as a vector with real and imaginary parts separate
    % %%matrl
    result = [real(result(rad_idx_dbar)); imag(result(rad_idx_dbar))];
end

function test_d_bar
    [fmdl, inv_model] = mk_dbar_model(16);
    
    img = mk_image(fmdl, 1);
    vh = fwd_solve(img);
    img_inh = set_test_target(img);
    vi = fwd_solve(img_inh);
    
    img_solved = inv_solve(inv_model, vi, vh);
    show_fem(img_solved);
end

function img_out = set_test_target(img)
    cog = interp_mesh(img.fwd_model, 0);
    tar_offset = 0.6;
    cog(:,1,:) = cog(:,1,:) - tar_offset;
    tar_rad = 0.3;
    idx_tar = cog(:,1,:).^2 + cog(:,2,:).^2 < tar_rad.^2;
    img.elem_data(idx_tar) = img.elem_data(1)*1.2;
    img_out = img;
end
