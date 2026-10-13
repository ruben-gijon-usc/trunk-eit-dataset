function [cmdl, c2f]= mk_grid_model(fmdl, xvec, yvec, zvec, removefn);
% MK_GRID_MODEL: Create reconstruction model on pixelated grid 
%  [cmdl,coarse2fine]= mk_grid_model(fmdl, xvec, yvec, zvec, removefn);
%
% Outputs:
%  cmdl - eidors reconstruction model (coarse model)
%  coarse2fine - c2f mapping to put onto fmdl (specify [] to not use)
%
% Inputs:
%  fmdl - fine model (forward model) to create coarse2fine mapping
%  xvec - x edges
%  yvec - y edges
%  zvec - z edges (optional - to create 3D model)
%  removefn(xyzc) - function which elements to remove, given xyzctr
%  removefn = 'outside' => remove elements outside the fmdl
%    e.g. removefn = @(xyz) vecnorm(xyz(:,1:2),2,2)>1.0;
%
% if fmdl == [], then just create the grid model without c2f
%
% Example: for circular model
%  grid= linspace(-2,2,20);     % x,y grid
%  [gmdl]= mk_grid_model([],grid,grid,[], @(xyz) vecnorm(xyz,2,2)>1);
%
% Example: for constructing an inverse model
%  grid{1}= linspace(-2,2,20);     % x grid
%  grid{2}= linspace(-0.5,+0.5,5); % y grid
%  grid{3}= linspace(-2, 0,20);    % z grid
%  imdl = select_imdl( fmdl, {'Basic GN dif'});
%  [imdl.rec_model,imdl.fwd_model.coarse2fine]= mk_grid_model(fmdl,grid{:});
%
% See also MK_COARSE_FINE_MAPPING, MK_PIXEL_SLICE

% ISSUES:
%  Ensure that grids are defined from smallest to largest

% (C) 2008 Andy Adler. License: GPL version 2 or version 3
% $Id: mk_grid_model.m 7044 2024-11-30 15:53:22Z aadler $

if ischar(fmdl) && strcmp(fmdl,'UNIT_TEST'); do_unit_test; return; end


if nargin<5 % removefn not provided
   removefn = [];
end
if nargin == 3 || isempty(zvec)
   do3d= false;
elseif nargin > 3
   do3d=true;
else
   error('check nargin');
end

if ~do3d
   [cmdl,rmelems] = mk_2d_grid(xvec,yvec,removefn);
else
   [cmdl,rmelems] = mk_3d_grid(xvec,yvec,zvec,removefn);
end

% this had too many side effects
cmdl = set_pixel_pos(cmdl,xvec,yvec);% same for 2d and 3d

% put in the centre (or near it)
ctr = ones(num_nodes(cmdl),1)*mean(cmdl.nodes);
dctr= sum( (cmdl.nodes - ctr).^2, 2);
[jnk, c_idx] = min(dctr);
cmdl.gnd_node = c_idx(1);

if ~isempty( fmdl)
    if size(fmdl.nodes,2) == 2
        assert(~do3d);
        c2f= calc_c2f_2d( fmdl, xvec, yvec);
        
    else
        if ~do3d
            % here we could incorporate z_depth
            zvec = [ min(fmdl.nodes(:,3)) - 1; max(fmdl.nodes(:,3))+1 ];
            tmp = mk_3d_grid(xvec,yvec,zvec,removefn);
        else
            tmp = cmdl;
        end
        c2f = mk_grid_c2f(fmdl,tmp);
    end
else
    c2f = [];
end

% now remove elements outsize shape
if ischar(rmelems{2}) && strcmp(rmelems{2},'OUTSIDE')
   rmelems{2} = (sum(c2f,1) < eps);
   rmelems{1} = logical(kron(rmelems{2}, true(1,rmelems{1})));
end
if ~isempty(rmelems{1})
   cmdl.elems( rmelems{1}, : ) = [];
   cmdl.coarse2fine( rmelems{1},     :      ) = [];
   cmdl.coarse2fine(    :      , rmelems{2} ) = [];
   if ~isempty(c2f)
      c2f(:,rmelems{2}) = [];
   end
%  cmdl.remover = rmelems; % debugging
end

cmdl.boundary = find_boundary( cmdl.elems);

cmdl = mdl_normalize(cmdl, 'default');
cmdl.solve =      @eidors_default;
cmdl.system_mat = @eidors_default;
cmdl.jacobian   = @eidors_default;

% standard field order
cmdl = eidors_obj('set', cmdl);

function c2f= calc_c2f_2d( fmdl, xvec, yvec);
   nef= size( fmdl.elems,1);
   c2f= sparse(nef,0);
   mdl_pts = interp_mesh( fmdl, 3);
   x_pts = squeeze(mdl_pts(:,1,:));
   y_pts = squeeze(mdl_pts(:,2,:));
   for yi= 1:length(yvec)-1
         in_y_pts = y_pts >= yvec(yi) & y_pts < yvec(yi+1);
      for xi= 1:length(xvec)-1
          in_x_pts =  x_pts >= xvec(xi) & x_pts < xvec(xi+1);
          in_pts = mean( in_y_pts & in_x_pts , 2);
          c2f = [c2f,sparse(in_pts)];
      end
   end

% Old implementation, replaced by mk_grid_c2f   
function c2f= calc_c2f_3d( fmdl, xvec, yvec, zvec);
%  c2f= mk_coarse_fine_mapping( fmdl, cmdl);
   nef= size( fmdl.elems,1);
%  c2f= sparse(nef,0);
   c2fiidx= [];
   c2fjidx= [];
   c2fdata= [];
   jidx= 0;
   mdl_pts = interp_mesh( fmdl, 3);
   x_pts = squeeze(mdl_pts(:,1,:));
   y_pts = squeeze(mdl_pts(:,2,:));
   z_pts = squeeze(mdl_pts(:,3,:));
   
   in_x_pts = calc_in_d_pts( x_pts, xvec);
   in_y_pts = calc_in_d_pts( y_pts, yvec);
   in_z_pts = calc_in_d_pts( z_pts, zvec);

   for zi= 1:length(zvec)-1
      for yi= 1:length(yvec)-1
             in_yz_pts = in_y_pts{yi} & in_z_pts{zi};
         for xi= 1:length(xvec)-1
             in_pts = mean( in_x_pts{xi} & in_yz_pts, 2);
             % c2f = [c2f,sparse(in_pts)];
             [ii,jj,vv] = find(in_pts);
             c2fiidx= [c2fiidx;ii];
             c2fjidx= [c2fjidx;jj+jidx]; jidx=jidx+1;
             c2fdata= [c2fdata;vv];
         end
      end
   end
   c2f= sparse(c2fiidx,c2fjidx,c2fdata, length(in_pts), jidx);

function [cmdl,rmelems]= mk_2d_grid(xvec, yvec, removefn);
   xlen = length(xvec);
   ylen = length(yvec);
   cmdl= eidors_obj('fwd_model', ...
            sprintf('Grid model %d x %d', xlen, ylen) );

   [x,y]= ndgrid( xvec, yvec);
   cmdl.nodes= [x(:),y(:)];
   k= 1:xlen-1;
   elem_frac = [ k;k+1;k+xlen; ...
                 k+1;k+xlen;k+xlen+1];
   elem_frac= reshape(elem_frac, 3,[])';
   cmdl.elems=  [];
   for j=0:ylen-2
      cmdl.elems=  [cmdl.elems; elem_frac + xlen*j];
   end

   if isempty(removefn)
      rmelems{1} = [];
      rmelems{2} = [];
   elseif ischar(removefn) && strcmp(upper(removefn),'OUTSIDE') 
      rmelems{2} = 'OUTSIDE'; % solve later
      rmelems{1} = 2; % divide factor
   else % it's a function
      % Ensure we get centre of each grid square from triangles
      xyzc = interp_mesh(cmdl);
      xyzc = kron(xyzc(1:2:end,:) + xyzc(2:2:end,:),[1;1]/2);
      rmelems{1} = removefn(xyzc);
      rmelems{2} = rmelems{1}(1:2:end);
   end

% assign one single parameter to each square element
   e= size(cmdl.elems,1);
   params= ceil(( 1:e )/2);
   cmdl.coarse2fine = sparse(1:e,params,1,e,max(params));

   cmdl.boundary = find_boundary( cmdl.elems);

function [cmdl,rmelems]= mk_3d_grid(xvec, yvec, zvec,removefn)
   xlen = length(xvec);
   ylen = length(yvec);
   zlen = length(zvec);
   if zlen<2 || ylen<2 || xlen<2
      error('Need at least 2 components for each gridpoint')
   end
   cmdl= eidors_obj('fwd_model', ...
            sprintf('Grid model %d x %d x %d', xlen, ylen, zlen) );

   [x,y,z]= ndgrid( xvec, yvec, zvec);
   cmdl.nodes= [x(:),y(:),z(:)];
   k= 1:xlen-1;
   ac = xlen; up = xlen*ylen; % accross vs up
   elem_frac = [ k;     k+1 ;  k+ac;   k+up;  ...
                 k+1;   k+ac;  k+up;   k+up+1; ...
                 k+ac;  k+up;  k+up+1; k+up+ac; ...
                 k+1;   k+ac;  k+ac+1; k+up+1; ...
                 k+ac;  k+ac+1;k+up+1; k+up+ac; ...
                 k+ac+1;k+up+1;k+up+ac;k+up+ac+1];
   elem_frac= reshape(elem_frac, 4,[])';
   sz_elem_frac = size(elem_frac);
   row_frac =  zeros(sz_elem_frac .* [ylen-1,1]);
   for j=0:ylen-2
      idx = (1:sz_elem_frac(1)) + j*sz_elem_frac(1);
      row_frac(idx,:)=  elem_frac + ac*j;
   end
   
   sz_row_frac = size(row_frac);
   cmdl.elems=  zeros(sz_row_frac .* [zlen-1,1]);
   for k=0:zlen-2
      idx = (1:sz_row_frac(1)) + k*sz_row_frac(1);
      cmdl.elems(idx,:) =  row_frac + up*k;
   end

   if isempty(removefn)
      rmelems{1} = [];
      rmelems{2} = [];
   elseif ischar(removefn) && strcmp(upper(removefn),'OUTSIDE') 
      rmelems{2} = 'OUTSIDE'; % solve later
      rmelems{1} = 6; % divide factor
   else % it's a function
      % Ensure we get centre of each grid square from triangles
      xyzc = interp_mesh(cmdl);
      xyzc = kron(xyzc(1:6:end,:) + ...
                  xyzc(2:6:end,:) + ...
                  xyzc(3:6:end,:) + ...
                  xyzc(4:6:end,:) + ...
                  xyzc(5:6:end,:) + ...
                  xyzc(6:6:end,:),[1;1;1;1;1;1]/6);
      rmelems{1} = removefn(xyzc);
      rmelems{2} = rmelems{1}(1:6:end);
   end

% assign one single parameter to each square element
   e= size(cmdl.elems,1);
   params= ceil(( 1:e )/6);
   cmdl.coarse2fine = sparse(1:e,params,1,e,max(params));

function mdl = set_pixel_pos(mdl, xvec, yvec)
   x = xvec(1:end-1) + 0.5*diff(xvec);
   y = yvec(1:end-1) + 0.5*diff(yvec);
   y = y(end:-1:1); %get the medical orientation right
   mdl.mdl_slice_mapper.x_pts = x;
   mdl.mdl_slice_mapper.y_pts = y;
   
   
function in_d_pts = calc_in_d_pts( d_pts, dvec);
   l1dvec= length(dvec)-1;
   in_d_pts = cell(l1dvec,1);
   for i= 1:l1dvec
      in_d_pts{i} = d_pts >= dvec(i) & d_pts < dvec(i+1);
   end

function do_unit_test
imdl = mk_common_model('b2c2',16); imdl.hyperparameter.value = 1e-3;
img = mk_image(imdl,1);     vh= fwd_solve(img);
img.elem_data([51,23])=1.1; vi= fwd_solve(img);
subplot(3,4,1); show_fem(img);
subplot(3,4,2); show_fem(inv_solve(imdl, vh, vi));

grid = linspace(-1,1,33);
[imdl.rec_model, imdl.fwd_model.coarse2fine] = ...
     mk_grid_model( imdl.fwd_model, grid, grid );
subplot(3,4,3); show_fem(inv_solve(imdl, vh, vi));
hold on; hh=show_fem(img); set(hh,'FaceAlpha',0,'EdgeColor',[0,0,1]); hold off;

if 0
  outside = find(sum(imdl.fwd_model.coarse2fine,1) < eps);
  imdl.fwd_model.coarse2fine(:,outside) = [];
  imdl.rec_model.coarse2fine(:,outside) = [];
  rec_out = [2*outside-1,2*outside];
  imdl.rec_model.coarse2fine(rec_out,:) = [];
  imdl.rec_model.elems(rec_out,:) = [];
else
  rmfn = 'outside';
  [imdl.rec_model, imdl.fwd_model.coarse2fine] = ...
       mk_grid_model( imdl.fwd_model, grid, grid, [],rmfn );
end
subplot(3,4,4); show_fem(inv_solve(imdl, vh, vi));
hold on; hh=show_fem(img); set(hh,'FaceAlpha',0,'EdgeColor',[0,0,1]); hold off;

rmfn = @(xyz) vecnorm(xyz,2,2)>1;
[imdl.rec_model, imdl.fwd_model.coarse2fine] = ...
     mk_grid_model( imdl.fwd_model, grid, grid, [],rmfn );
subplot(3,4,5); show_fem(inv_solve(imdl, vh, vi));
hold on; hh=show_fem(img); set(hh,'FaceAlpha',0,'EdgeColor',[0,0,1]); hold off;

subplot(3,4,6)
imdl = mk_common_model('n3r2',[16,2]);
A= [390 391 393 396 402 478 479 480 484 486 664 665 666 667 668 670 671 672 676 677 678 755 760 761];
B= [318 319 321 324 330 439 440 441 445 447 592 593 594 595 596 598 599 600 604 605 606 716 721 722];
img = mk_image(imdl.fwd_model,1); vh = fwd_solve(img);
img.elem_data(A)=1.5; img.elem_data(B)=0.5; vi=fwd_solve(img);
show_fem(img);

subplot(3,4,7)
[cmdl, c2f]= mk_grid_model(img.fwd_model, -.8:.1:.8, -.8:.1:.8, 0:.5:3);
v = c2f'*(img.elem_data-1) + 1;
img2 = mk_image(cmdl, v);
show_fem(img2)

subplot(3,4,8)
rmfn = @(xyz) vecnorm(xyz(:,1:2),2,2)>1.0;
[imdl.rec_model, imdl.fwd_model.coarse2fine] = ...
    mk_grid_model(img.fwd_model, -1:.1:1, -1:.1:1, 0:.5:3,rmfn );
%v = imdl.fwd_model.coarse2fine'*(img.elem_data-1)+1;
%img2 = mk_image(imdl.rec_model, v);
img2= inv_solve(imdl,vh,vi);
show_fem(img2)

subplot(3,4,9)
[imdl.rec_model, imdl.fwd_model.coarse2fine] = ...
    mk_grid_model(img.fwd_model, -1:.1:1, -1:.1:1, 0:.5:3,'outside');
%v = imdl.fwd_model.coarse2fine'*(img.elem_data-1)+1;
%img2 = mk_image(imdl.rec_model, v);
img2= inv_solve(imdl,vh,vi);
show_fem(img2)

subplot(3,4,10)
   l48 = linspace(0.5,48.5,24+1); 
   l32 =-linspace(0.5,32.5,16+1) + 20; 
   rmfn = @(xyz) vecnorm(xyz,2,2)>40;
   fmdl = mk_grid_model([],l48,l32,l48,rmfn);
show_fem(fmdl)
