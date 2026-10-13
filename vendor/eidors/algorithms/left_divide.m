function [V] = left_divide(E,I,tol,~,V)
%[V] = LEFT_DIVIDE(E,I,tol,pp,V);
%[V] = LEFT_DIVIDE(E,I,fmdl)
% 
% Implements left division for symmetric positive definite system solves
% such as the sparse forward solve and dense solve for a GN descent
% direction. LEFT_DIVIDE is optimised for symmetric matrices and overcomes
% small inefficiencies of matlab's mldivide. For non-symmetric solves 
% please use mldivide.
%
% Also uses conjugate gradients (for large problems).
%
% E   = The full rank system matrix
% I   = The currents matrix (RHS)
% tol = The tolerance in the forward solution, e.g. 1e-5
%
% pp,V are old options from previous solver. tilde used in arguments list
% to ignore pp and keep matlab's code analyzer happy

% (c) N. Polydorides 2003 % Copying permitted under terms of GNU GPL
% $Id: left_divide.m 7141 2024-12-29 23:26:27Z aadler $

if ischar(E) && strcmp(E,'UNIT_TEST'); do_unit_test; return; end

if nargin<3;
   tol=1e-8;
end
do_pcg = false;
if isstruct(tol);
   fmdl = tol;
   try
      do_pcg = fmdl.left_divide.do_pcg;
   catch
   end
   try 
      tol = fmdl.left_divide.tol;
   catch
      tol = 1e-8;
   end
   try 
      V = fmdl.left_divide.V_initial;
   catch
      sz= [size(E),size(I)];
      V = eidors_obj('get-cache', sz, 'left_divide_V');
      if isempty(V); V = zeros(size(E,1),size(I,2)); end
   end
end

if ~do_pcg
   try
     V= non_iterative(E,I);
   catch excp
       % TODO: check if this catch block is needed
       if ~strcmp(excp.identifier , 'MATLAB:nomem')
           rethrow(excp); % rethrow error
       end
       
       eidors_msg('Memory exhausted for inverse. Trying PCG',2);
       V=iterative_solve(E,I,tol,V,fmdl);
   end
else
   V=iterative_solve(E,I,tol,V,fmdl);
end

function V= non_iterative(E,I);
    % V= E\I;
    % This takes MUCH longer when you have  more vectors in I,
    %  even if they are repeated. There must be some way to simplify
    %  this to speed it up. Matlab's sparse operators really should
    %  do this for you.
    
    % TODO: 
    % 1. change from QR implementation to basis implementation
    % 2. implement selection for required nodal values
    % 3. cache basis solve
    % 4. possibly change to itterative for successive solves on the same
    %    mesh
    if issparse(E)
        
        inotzeros = logical(any(I,2));
        % Don't need to cache, this is fast
        [Qi,R,p] = qr(I(inotzeros,:),'vector'); % use permution to reduce nnz
        rnotzeros = logical(any(R,2));
        R= R(rnotzeros,:);
        Q = zeros(size(I,1), size(R,1)); % Faster if not sparse
        Q(inotzeros,:) = Qi(:,rnotzeros);
        %disp([size(I), size(Qi), size(R)])
%        [Q,R] = qr(I,0);
%        rnotzeros = any(R~=0,2);
%        Q= Q(:,rnotzeros);
%        R= R(rnotzeros,:);

%       Conditioning of solution
% TODO: Figure out when we can do preconditioning -- plan for 3.13
        if 1 %no conditioning
            V= (E \ Q)*R;  %% OLD
        else
            P= spdiag(sqrt(1./diag(E)));
            V= P*((P*E*P) \ (P*Q))*R;
        end
        V(:,p) = V; % undo the permutation
        
    else
        if isreal(E)
            try
                % for dense solve of tikhonov regularised least squares
                % matrix E is symmetric if it is of the form
                % (J.'*W*J + hp^2*R.'R) and is real
                opts.SYM=true;
% TODO: refactor this for the complex case
                opts.POSDEF=true;

                if 0 % conditioning
                    V= linsolve(E,I,opts);
                else
% Matlab does bad things with memory with sparse x full.
%  Instead use .* 
%                   P= spdiag(sqrt(1./diag(E)));
%                   P_ = abs(P)>1e100; % prevent ridiculous values
                        % large values lead to non-pos-def chol
%                   P(P_) = 1e100 * sign(P(P_));
%                   EP = P*E*P; EP=0.5*(EP+EP'); %force symmetric
%                   V= P*linsolve(EP,P*I,opts);
                    Pd= sqrt(1./diag(E)); % must be +ve
                    Pd(Pd>1e100) = 1e100;
% In octave, bsxfun is more accutate than times. Not sure why
                    E  = bsxfun(@times, Pd,E);
                    E  = bsxfun(@times, E, Pd');
%                   E  = Pd .* E;
%                   E  = E .*(Pd');
%                   EP = (Pd.*E) .* (Pd'); % Stupid Matlab gets killed
                    E =0.5*(E + E'); %force symmetric
%                   V= Pd.*linsolve(E,Pd.*I,opts);
                    V= bsxfun(@times, Pd, linsolve(E, ...
                       bsxfun(@times, Pd, I) ,opts));
% TODO: many ways to improve memory handling. Maybe need mex file?
                     
                end
            catch Mexcp
                
                % error handling 
                if(strcmp(Mexcp.identifier,'MATLAB:posdef'))
                    warning('EIDORS:leftDivideSymmetry',...
                        ['left_divide is optimised for symmetric ',...
                        'positive definite matrices.']);
                else 
                    warning(['Error with linsolve in left_divide, trying backslash.\n',...
                        'Error identifier: ',Mexcp.identifier]);
                end
                
                % continue solve with backslash
                V=E\I;
            end
        else
            % cholesky only works for real valued system matrices
            V=E\I;
        end
    end
    
    % TODO: Iteratively refine
    %  From GH Scott: "once we have
    %   computed the approximate solution x, we perform one step
    %   of iterative refinement by computing the residual: r = Ax - b
    %   and then recalling the solve routine to solve
    %   Adx = r for the correction dx.
    % However, we don't want to repeat the '\', so we implement
    %   the underlying algorithm:
    %   If A is sparse, then MATLAB software uses CHOLMOD (after 7.2) to compute X.
    %    The computations result in  P'*A*P = R'*R
    %   where P is a permutation matrix generated by amd, and R is
    %   an upper triangular matrix. In this case, X = P*(R\(R'\(P'*B)))
    %
    % See also:
    % http://www.cs.berkeley.edu/~wkahan/MxMulEps.pdf
    % especially page 15 where it discusses the value of iterative refinement
    %  without extra precision bits.  ALso, we need to enable
    
function V=iterative_solve(E,I,tol,V,fmdl)
    
    [n_nodes,n_stims] = size(I);
    if isreal(E)
        opts.droptol = tol*100;
        opts.type = 'ict';
        U = ichol(E, opts);
        L = U';
        cgsolver = @pcg;
    else %Complex
        opts.droptol = tol/10;
        [L,U] = ilu(E, opts);
        cgsolver = @bicgstab;
    end
    
    for i=1:n_stims
        [V(:,i),~] = feval( cgsolver, E,I(:,i), ...
            tol*norm(I(:,i)),n_nodes,L,U,V(:,i));
    end
    sz= [size(E),size(I)];
    eidors_obj('set-cache', sz, 'left_divide_V', V);
    

% Test code
function do_unit_test
% do_timing_unit_tests; return
conditioning_test;

% test solvers are unaffected 
inv_solve('UNIT_TEST')
fwd_solve_1st_order('UNIT_TEST')

% test non-symmetric handling
s=warning('QUERY','EIDORS:leftDivideSymmetry');
warning('OFF','EIDORS:leftDivideSymmetry')
lastwarn('')
A=rand(1e3);
b=rand(1e3);

left_divide(A,b);
[~, LASTID] = lastwarn;
unit_test_cmp('sym warn',LASTID,'EIDORS:leftDivideSymmetry')
warning(s);

% test dense sym posdef solve
imdl=mk_common_model('n3r2',[16,2]);
img=mk_image(imdl,1);
img.elem_data=1+0.1*rand(size(img.elem_data));
J   = calc_jacobian(img);
RtR = calc_RtR_prior(imdl);
W   = calc_meas_icov(imdl);
hp  = calc_hyperparameter(imdl);
LHS = (J'*W*J +  hp^2*RtR);
RHS = J'*W;
unit_test_cmp('dense chol',LHS\RHS,left_divide(LHS,RHS),1e-13)

% Test if left divide correctly conditions
function conditioning_test
    Y = calc_jacobian(mk_image(mk_common_model('d2c2',16)));
    Sn = randn(208); Sn=Sn*Sn'; %speye(N_meas) * opt.noise_covar; % Noise covariance
    PJt= Y'; noiselev = 1;
    Sn(140,140)= 1e20;
    M  = (Y*Y' + noiselev^2*Sn);
%   subplot(211); semilogy(diag(M));
    RM = (M\(PJt'))';    %PJt/M;
%   subplot(212); semilogy(diag(M));
    RM2=  left_divide(M',PJt')';    %PJt/M;


    unit_test_cmp('Conditioning',RM,RM2,1e-11)


function do_empty_c2f_test
% code from dual_model/centre_slice --- gets warning
   n_sims= 20;
   stim = mk_stim_patterns(16,2,'{ad}','{ad}',{},1);
   fmdl = mk_library_model('cylinder_16x2el_vfine');
   fmdl.stimulation = stim;
   [vh,vi,xyzr_pt]= simulate_3d_movement( n_sims, fmdl);

   % Create and show inverse solver
   imdl = mk_common_model('b3cr',[16,2]);

   f_mdl = mk_library_model('cylinder_16x2el_coarse');
   f_mdl.stimulation = stim;
   imdl.fwd_model = f_mdl;

   imdl2d= mk_common_model('b2c2',16);
   c_mdl= imdl2d.fwd_model;

   imdl.rec_model= c_mdl;
   c2f= mk_coarse_fine_mapping( f_mdl, c_mdl);
   imdl.RtR_prior = @prior_gaussian_HPF;
   imdl.solve = @inv_solve_diff_GN_one_step;
   imdl.hyperparameter.value= 0.1;


   imdl.hyperparameter.value= 0.05; scl= 15;

   c_mdl.mk_coarse_fine_mapping.f2c_offset = [0,0,(1-.3)*scl];
   c_mdl.mk_coarse_fine_mapping.z_depth = 0.1;
   c2f= mk_coarse_fine_mapping( f_mdl, c_mdl);
   imdl.fwd_model.coarse2fine = c2f;
   imgc0= inv_solve(imdl, vh, vi);
   % Show image of reconstruction in upper planes% show_slices(imgc0);

function do_timing_unit_tests
% The point of these tests are to verify which 
%  matrices should be sparse and which full. 
% Conclusion is that Q should be full - AA (jun 2022)

%eidors_cache off
Nel = 64;
for mn = 1:4; switch mn
    case 1;
        fmdl = mk_common_model('d2c0',64);
        fmdl = fmdl.fwd_model;
    case 2;
        fmdl = mk_common_model('l2c0',64);
        fmdl = fmdl.fwd_model;
    case 3;
        fmdl = mk_circ_tank(16, linspace(-1,1,41), {'planes',Nel, 21});
        fmdl.solve = @eidors_default;
        fmdl.system_mat = @eidors_default;
    case 4;
        fmdl = mk_circ_tank(32, linspace(-1,1,41), {'planes',Nel, 21});
        fmdl.solve = @eidors_default;
        fmdl.system_mat = @eidors_default;
    end
    stim= mk_stim_patterns(Nel,1,[0,3],[0,3],{},1);
    for sp = 1:2; switch sp;
       case 1; 
           fmdl.stimulation = stim; 
       case 2;
            SSMM = stim_meas_list(stim);
            [~,idx] = sort(rand(size(SSMM,1),1));
            fmdl.stimulation = stim_meas_list(SSMM(idx,:)); 
        end;
        img = mk_image(fmdl,1);
%       fwd_solve(img);
        s_mat = calc_system_mat(img);
        idx= 1:size(s_mat.E,1);
        idx(img.fwd_model.gnd_node) = [];
        E = s_mat.E(idx,idx);
        pp= fwd_model_parameters( img.fwd_model, 'skip_VOLUME' );
        I = pp.QQ(idx,:); 

        inotzeros = logical(any(I,2));
        [Qi,R] = qr(I(inotzeros,:),0);
        rnotzeros = logical(any(R,2));
        R= R(rnotzeros,:);
        Q = sparse(size(I,1), size(R,1));
        Q(inotzeros,:) = Qi(:,rnotzeros);
        t=[];
        tic; T = E \ Q; t(end+1) = toc;
        tic; V = T * R; t(end+1) = toc;
        tic; T = full(E \ Q); t(end+1) = toc;
        tic; V = T * R; t(end+1) = toc;
        tic; T = E \ full(Q); t(end+1) = toc;
        tic; V = T * R; t(end+1) = toc;
        disp(t)
    end
end
