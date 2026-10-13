function out = mk_head_model_adult(varargin)
%MK_HEAD_MODEL_ADULT Adult head model 
%
% MK_HEAD_MODEL_ADULT provides predefined and custom head FEMs based on
% models contributed by Andrew Tizzard. You may be asked to download
% missing files.
%
% FMDL = MK_HEAD_MODEL_ADULT() 
% Returns a head fwd_model with no electrodes or tissues
%
% FMDL = MK_HEAD_MODEL_ADULT(elec_pos, elec_shape)
% FMDL = MK_HEAD_MODEL_ADULT(elec_pos, elec_shape, maxh)
% Allows specifying electrode configuration, see PLACE_ELEC_ON_SURF.
% Default maxh = 8.
%
% FMDL = MK_HEAD_MODEL_ADULT(elecstr)
% A predefined model with round electrodes (radius 2 mm). 
%    '10-10'         - 73 labeled electrodes (10-10 placement system),
%                      no stimulation.
%    '2x16_planar'   - 2 rings of 16 electrodes, planar stimulation
%    '2x16_odd-even' - 2 rings, each stimulation is cross-plane 
%    '2x16_square'   - 2 rings, every second stimulation is cross-plane
%    '2x16_adjacent' - like square, but with adjacent stimulation
%    '1x32_ring'     - single ring of 32 electrodes
%    'no_elecs'      - no electrodes
%
% IMG = MK_HEAD_MODEL_ADULT(... , tissuespec)
% Returns a head model containing separate regions defining scalp, skull, 
% CSF and brain as an image structure with conductivitity values. 
% TISSUESPEC must be one of:
%    'coarse'        - (defualt) tissue surfaces with maxh = 5mm
%    'fine'          - tissue surfaces with maxh = 2mm
%    'void'          - no internal tissues (void), useful for testing
%                      electrode positions
%
% MK_HEAD_MODEL('show_10-10') creates a figure showing the positions and
% labels of the 10-10 electrodes.
%
% 
% CITATION_REQUEST:
% TITLE: FEM Electrode Refinement for Electrical Impedance Tomography 
% AUTHOR: B Grychtol and A Adler
% JOURNAL: Engineering in Medicine and Biology Society (EMBC), 2013 Annual 
% International Conference of the IEEE 
% YEAR: 2013
% DOI: 10.1109/EMBC.2013.6611026
%
% CITATION_REQUEST:
% TITLE: Improving the Finite Element Forward Model of the Human Head by 
% Warping using Elastic Deformation
% AUTHOR: A Tizzard and RH Bayford
% JOURNAL: Physiol Meas
% YEAR: 2007
% VOL: 28
% PAGE: S163-S182
% DOI: 10.1088/0967-3334/28/7/S13
%
% See also PLAC_ELEC_ON_SURF

% (C) 2012-2024 Bartek Grychtol. License: GPL v2 or v3
% $Id: mk_head_model_adult.m 7031 2024-11-28 21:12:36Z bgrychtol $

citeme(mfilename)

if nargin==1 && ischar(varargin{1}) && strcmp(varargin{1},'UNIT_TEST')
   do_unit_test; return;
end

if nargin == 0
   out = get_base_model; % Returns a head fwd_model with no electrodes or tissues
end

copyright = '';
organspec = 'coarse';
if nargin > 1 && ischar(varargin{end})
   organspec = varargin{end};
   varargin(end) = [];
end

switch length(varargin)
   case 0
      % nothing
   case 1
      if ischar(varargin{1})
         if strcmp(varargin{1},'show_10-10')
            [out, img, ~] = get_base_model;
            get_elec_pos(img, true);
            return
         end
         copyright = '(C) EIDORS Project';
         out = build_predef_elecs(varargin{1}, organspec);
      else
         error('EIDORS:WrongInput','Electrode specification not understood');
      end
   case {2, 3}
      [out, img, materials] = get_base_model;
      out = build_head_elecs(out, varargin{:});
      out = mk_shell_image(img, materials, out, organspec);

   otherwise
      error('EIDORS:WrongInput', 'Too many inputs');
end

out = eidors_obj('set', out); % impose standard field order

if ~isempty(copyright)
   out.copyright = copyright;
end

out.attribution = ...
   ['Modified from "SAH262" by Andrew Tizzard '...
    '(Tizzard, A. and Bayford, R.H. (2007). Improving the Finite Element ' ...
    'Forward Model of the Human Head by Warping using Elastic Deformation. '...
    'Physiol.Meas. 28:S163-S182)'];

if strcmp(out.type, 'image')
   try out.fwd_model.copyright = out.copyright; end
   out.fwd_model.attribution = out.attribution;
end

ws = warning('off', 'backtrace');
warning('EIDORS:License', ...
   ['The head model is a derivative of worked licensed under CC BY.\n' ...
    'Please make sure to attribute correctly. ' ...
    'See the attribution field for details.']);
warning(ws.state, 'backtrace');

function out = build_predef_elecs(stimpat, organspec)
names = [];
switch stimpat
   case {'10-10', 'no_elecs'}
      mdl_name = stimpat;
   case {'2x16_planar'; '2x16_odd-even'; '2x16_square';
         '2x16_adjacent';'1x32_ring'}
      mdl_name = '3_rings';
   otherwise
      error('EIDORS:WrongInput', 'Don''t understand model %s.', stimpat )
end

cache_path = get_cache_path;
fname = sprintf('head_model_%s',mdl_name);
suffix = '';
if ~isempty(organspec)
   fname = [fname '_' organspec];
   suffix = [' - ' organspec];
end
name = ['Adult Head Model' suffix ' - ' stimpat];

fpath = [cache_path filesep fname '.mat'];
if exist(fpath,'file')
   eidors_msg('@@ Using stored model.', 3);
   load(fpath, 'out');
else
   [out, img, materials] = get_base_model;
   switch stimpat
      case 'no_elecs'
         elec_pos = [];
      case '10-10'
         [elec_pos, names] = get_elec_pos(img);
         elec_spec = [2 0 0.2];
         maxh = 8;
      case {'2x16_planar'; '2x16_odd-even'; '2x16_square';
            '2x16_adjacent';'1x32_ring'}
         Nel = 32;
         eth32 = -pi/2 : 2*pi/Nel : 2*pi-pi/2; eth32(1)=[];
         eth16 = eth32(1:2:end);
         eth32 = eth32 - pi/Nel;
         t = [eth16, eth32, eth16];
         xm=0;
         ym=12;
         a=80; b=95;
         elec_pos(:,1) = xm+a*cos(t);
         elec_pos(:,2) = ym+b*sin(t);
         elec_pos(1:16,3) = 0;
         elec_pos(17:48,3) = 5;
         elec_pos(49:64,3) = 10;         
         elec_spec = [2 0 0.2];
         maxh = 8;
      otherwise
         error('EIDORS:WrongInput','Parameter electsr not understood');
   end
   if ~isempty(elec_pos)
      out = build_head_elecs(out, elec_pos, elec_spec, maxh);
   end
   out.name = name;
   if ~isempty(names)
      for i = 1:length(out.electrode)
         out.electrode(i).label = names{i};
      end
   end
   out = mk_shell_image(img, materials, out, organspec);
   save(fpath,'out');

end

out.name = name;

if strcmp(mdl_name,'3_rings')
   out = set_predef_stim(out, stimpat);
end

function out = mk_shell_image(img, materials, out, organspec)
if isempty(organspec), return, end
switch organspec
   case 'void'
      return
   case 'coarse'
      maxh = 5;
   case 'fine'
      maxh = 2;
   otherwise
      error('EIDORS:WrongInput', 'Parameter tissuespec not understood.');
end
[CSF, brain, skull2, bn2] = build_brain_CSF(img,materials,maxh);

% extract the outer skull surface from the scalp model with electrodes
cond = @(xyz) xyz(:,3) < -121.1 & ... 
              xyz(:,3) < (abs(xyz(:,2).^1/11) -123); 
[skull1, ~, bn1] = split_surface(out, cond);
skull1.elems(:,2:3) = skull1.elems(:,[3 2]); % flip normals

% close surfaces from the bottom and create volume meshes
p = bottom_z_from_xy ([bn1; bn2]); 
skull = build_skull(skull1, skull2, p, maxh);

% put it all together one by one
out.mat_idx = {1:length(out.elems)};
l1 = length(out.boundary);
out.boundary_numbers = uint32(ones(l1,1));
out = merge_meshes(out,skull,0.05);
l1 = length(out.boundary);
out.boundary_numbers(end+1:l1) = 2;
out = merge_meshes(out,CSF,0.025);
l1 = length(out.boundary);
out.boundary_numbers(end+1:l1) = 3;
out = merge_meshes(out,brain);
l1 = length(out.boundary);
out.boundary_numbers(end+1:l1) = 4;
d = sqrt(sum(out.nodes.^2,2));
[~, out.gnd_node] = min(d);
mats = [3 1 4 2];
out.mat_names = materials.names(mats);
out = mk_image(out,.5);

for i = 1:4
   out.elem_data(out.fwd_model.mat_idx{i}) = materials.sigma(mats(i));
end
% show_fem(out)
out.name = 'Head image with conductivities';

function [CSF, brain, skull2, bn2] = build_brain_CSF(img,materials,maxh)

cache_path = get_cache_path;
fname = sprintf('brain_C2F_maxh=%.1f.mat', maxh);
fpath = [cache_path filesep fname];
if exist(fpath, 'file')
   eidors_msg('@@ Using stored internal meshes.', 3);
   load(fpath, 'CSF', 'brain', 'skull2', 'bn2');
   return
end

idx = find(strcmp(materials.names, 'CSF'));
CSF = img.fwd_model; CSF.elems = CSF.elems(materials.ref==idx,:);
CSF = remesh_CSF(CSF, maxh);

% extract individual surfaces from the CSF model
cond = @(xyz) xyz(:,3) < -122.8; % condition to apply to the *rotated* model
[skull2, brain, bn2] = split_surface(CSF, cond);
skull2.elems(:,2:3) = skull2.elems(:,[3 2]); % flip normals
brain.elems(:,2:3) = brain.elems(:,[3 2]); % flip normals

% close surface from the bottom and create volume meshe
p = bottom_z_from_xy (bn2); 
brain = build_brain(brain,p);

save(fpath, 'CSF', 'brain', 'skull2', 'bn2');

function skull = build_skull(skull1, skull2, p, maxh)
% treat the bottom as an xy plane and mesh it
rz = rotation_matrix;
rot = rz*skull1.nodes'; rot = rot';
b1 = rot(:,3) < -121.1 & ...
     rot(:,3) < (abs(rot(:,2).^1/11) -123);
rot = rz*skull2.nodes'; rot = rot';
b2 = rot(:,3) < -122.7;
bn1 = skull1.nodes(b1,:);order_loop(bn1,1);
bn2 = skull2.nodes(b2,:);order_loop(bn2,2);
l1 = bn1(:,1:2); l1 = order_loop(l1,-1);
l2 = bn2(:,1:2); l2 = order_loop(l2,-1);
ng_write_opt('meshoptions.fineness',1);
bs = ng_mk_2d_model({l1, l2});
delete('ng.opt');
bs.nodes(:,3) = polyval(p,bs.nodes(:,2));
bs.elems(:,2:3) = bs.elems(:,[3 2]);
% merge surfaces and mesh the inside
skullsrf = merge_meshes(bs,skull1,skull2,min(maxh/5,1));
skull = gmsh_stl2tet(skullsrf,[],[],2);

function brain = build_brain(brain,p)

rz = rotation_matrix;
rot= rz*brain.nodes'; rot = rot';
bot = rot(:,3) < -123;
loop = order_loop(brain.nodes(bot,1:2),-1);
ng_write_opt('meshoptions.fineness',1);
bs = ng_mk_2d_model(loop);
delete('ng.opt');
bs.nodes(:,3) = polyval(p,bs.nodes(:,2));
bs.elems(:,2:3) = bs.elems(:,[3 2]); % reorient normals
mrgd = merge_meshes(brain,bs,0.6);
% stl_write(mrgd,'brain.stl');
brain = gmsh_stl2tet(mrgd,[],[],2);

function [upper, lower, bottom_nodes] = split_surface(mdl, fun)

% find indecies of the bottom surface nodes
rz = rotation_matrix;
rot= rz*mdl.nodes';
rot = rot';
bot = false(size(mdl.nodes,1),1);
bot(unique(mdl.boundary)) = true;
bot = bot & fun(rot);


bot_idx = find(bot);
on_boundary = ismember(bot_idx, unique(mdl.boundary));
bot = false(size(bot));
bot(bot_idx(on_boundary)) = true;

if eidors_debug('query','mk_head_model_adult:split_surface')
   clf
   jnk = mdl;
   jnk.nodes = rot;
   show_fem(jnk)
   crop_model([], @(x,y,z) z > -120)
   axis tight
   hold on
   plot3(rot(bot,1),rot(bot,2),rot(bot,3),'o');
   hold off
end

% remove the bottom surface from the boundary
b = mdl.boundary;
b = sum(bot(b),2)==3;
mdl.boundary(b,:) = [];
bottom_nodes = mdl.nodes(bot,:); % save bottom nodes for later

% separate surfaces
bnd = unique(mdl.boundary);
[jnk, pos] = min(abs(mdl.nodes(bnd,1))+abs(mdl.nodes(bnd,2))+mdl.nodes(bnd,3));
idx = find_connected_bnd_elems(mdl,bnd(pos));

% package
upper = build_shell(mdl, idx);
log_idx = true(size(mdl.boundary,1),1);
log_idx(idx) = false;
lower = build_shell(mdl, log_idx);

function shell = build_shell(mdl, idx)
shell.nodes = mdl.nodes;
shell.elems = mdl.boundary(idx,:);
shell.type = 'fwd_model';
shell = remove_unused_nodes(shell);
shell.boundary = shell.elems;

function rz = rotation_matrix
tz = -5.8/180 * pi;
rz = [1 0 0; 0 cos(tz) -sin(tz); 0 sin(tz) cos(tz) ;];

% find an equation relating y and z on the bottom surface
function p = bottom_z_from_xy (bn)
p = polyfit(bn(:,2),bn(:,3),6);
% clf
% plot(bn(:,2),bn(:,3),'.')
% hold on
% pp = polyval(p,-100:.1:100);
% plot(-100:.1:100,pp,'r')

function mdl = remesh_CSF(mdl, maxh)

opt.options.meshsize = maxh;
opt.options.curvaturesafety = 3.0;
opt.stloptions.yangle = 20.0;
opt.stloptions.contyangle = 50.0;
opt.stloptions.edgecornerangle = 100;
opt.stloptions.chartangle = 30;

mdl = fix_boundary(mdl);
mdl = ng_stl2tet(mdl, 'moderate', opt);

function idx = find_connected_bnd_elems(mdl,nd)
n_elem = length(mdl.boundary);
todo = false(1,n_elem);
done = false(1,n_elem);
tmp = false(1,length(mdl.nodes));
tmp(nd) = true;
% boundary elements that contain that node
id = any(tmp(mdl.boundary),2);
todo(id) = true;
mdl.elems = mdl.boundary;
mdl = fix_model(mdl, struct('face2edge',true));
f2e = mdl.face2edge;
i = repmat((1:length(mdl.boundary))',1,3);
S = sparse(i(:),f2e(:),true);
C = S*S'; % face2face connectivity matrix
C = logical(spdiags(zeros(n_elem,1),0,C)); % remove diagonal
while any(todo)
   [~, id] = find(C(todo,:));
   done(todo) = true;
   todo(id) = true;
   todo(done) = false;
end
idx = find(done);

function mdle = build_head_elecs(scalp, elec_pos, elec_spec, maxh)
if nargin < 4, maxh = 8; end
ng_opt = get_ng_opt(maxh);
mdle = place_elec_on_surf(scalp, elec_pos, elec_spec,ng_opt);
% fix orientation of the boundary
mdle = fix_boundary(mdle); mdl = mdle;

function [out, img, materials] = get_base_model()
[img, materials] = get_at_model();

cache_path = get_cache_path;
fname = 'base_model.mat';
fpath = [cache_path filesep fname];
if exist(fpath, 'file')
   eidors_msg('@@ Using stored base model.', 3);
   load(fpath, 'out');
   return
end

idx = find(strcmp(materials.names, 'scalp'));
scalp = img.fwd_model; scalp.elems = scalp.elems(materials.ref==idx,:);
out = fix_scalp(scalp); % re-meshes after merging duplicated surface nodes
save(fpath, 'out');

function mdl = fix_scalp(mdl)

%skull has some funny holes in the eyes.... fix manually
mdl = fix_model(mdl);
flip = mdl.elem2face(logical(mdl.boundary_face(mdl.elem2face).*mdl.inner_normal));
mdl.faces(flip,:) = mdl.faces(flip,[1 3 2]);
mdl.normals(flip,:) = -mdl.normals(flip,:);
mdl.boundary = mdl.faces(mdl.boundary_face,:);
mdl.elems = mdl.boundary;
% nodes 1065 and 1063 should be one
mdl.nodes(1063,:) = mean(mdl.nodes([1063 1065],:));
mdl.nodes(1065,:) = mean(mdl.nodes([109 111], :));
mdl.nodes(1045,:) = mean(mdl.nodes([1043 1045],:));
mdl.nodes(1043,:) = mean(mdl.nodes([101 103], :));

keep = any(reshape(mdl.nodes(mdl.elems,2) < -65 & ...
                    mdl.nodes(mdl.elems,2) > -75 & ...
                    mdl.nodes(mdl.elems,3) < -15 & ...
                    mdl.nodes(mdl.elems,3) > -30 & ...
                    mdl.nodes(mdl.elems,1) < -10 & ...
                    mdl.nodes(mdl.elems,1) > -20    ,[],3),2);
h63 = find(any(mdl.elems == 1063,2));
h65 = find(any(mdl.elems == 1065,2));
mdl.elems(h63(2),mdl.elems(h63(2),:)== 1063) = 1065;
mdl.elems(h63(4),mdl.elems(h63(4),:)== 1063) = 1065;
mdl.elems(h65(1),mdl.elems(h65(1),:)== 1065) = 1063;
mdl.elems(h65(3),mdl.elems(h65(3),:)== 1065) = 1063;
mdl.elems(h65(5),mdl.elems(h65(5),:)== 1065) = 1063;
h43 = find(any(mdl.elems == 1043,2));
h45 = find(any(mdl.elems == 1045,2));
mdl.elems(h43(1),mdl.elems(h43(1),:)== 1043) = 1045;
mdl.elems(h43(3),mdl.elems(h43(3),:)== 1043) = 1045;
mdl.elems(h43(5),mdl.elems(h43(5),:)== 1043) = 1045;
mdl.elems(h45(2),mdl.elems(h45(2),:)== 1045) = 1043;
mdl.elems(h45(4),mdl.elems(h45(4),:)== 1045) = 1043;

% mdl.boundary = mdl.elems(keep,:);
% clf
% show_fem(mdl);
% hold on

% we've only fixed the surface. Re-mesh the inside
mdl = gmsh_stl2tet(mdl,[],[],2);
mdl = fix_boundary(mdl);

function opt = get_ng_opt(maxh)
   opt.meshoptions.fineness = 6; % some options have no effect without this
   opt.options.curvaturesafety = 0.2;
   % small yangle preserves the original mesh, large encourages smoother
   % surface with nicer spreading of refinement
   opt.stloptions.yangle = 10;
 %    opt.stloptions.contyangle = 20;
   opt.stloptions.edgecornerangle = 0;
%    opt.stloptions.chartangle = 0;
   opt.stloptions.outerchartangle = 120;
   opt.stloptions.resthchartdistenable = 1;
   opt.stloptions.resthchartdistfac = 2.0; % encourages slower increase of element size
   opt.options.meshsize = maxh;
   opt.meshoptions.laststep = 'mv'; % don't need volume optimization
   opt.options.optsteps2d =  5; % but we can up surface optimization
   opt.options.badellimit = 120; % decrease the maximum allowed angle

function [img, materials] = get_at_model()

contrib = 'at-head-mesh'; file =  'SAH262.mat';
load(get_contrib_data(contrib,file),'tri', 'vtx', 'materials');

mdl = eidors_obj('fwd_model','sah262','nodes',vtx,'elems',tri);
img = mk_image(mdl,1);
for i = 1:4
   img.elem_data(materials.ref==i) = materials.sigma(i);
end

materials.names = cellstr(materials.names);

function out = get_cache_path
global eidors_objects
out = [eidors_objects.model_cache filesep 'head_model_adult'];
if ~exist(out, 'dir')
   mkdir(out);
end

%% ELECTRODE POSITIONS
function [e_ar, names] = get_elec_pos(mdl, show)
if nargin < 2, show = false; end
% cut through the nose:
slc = mdl_slice_mesher(mdl,[0 inf inf]);
slc.fwd_model.boundary = find_boundary(slc.fwd_model);

% exclude the nasal cavity
b = unique(slc.fwd_model.boundary);
box = [-87.6 -63.5;-88.2 -28;9.6  -26.8;11.6  -67.5];
b = b(~inpolygon(slc.fwd_model.nodes(b,2),slc.fwd_model.nodes(b,3),box(:,1),box(:,2)));
% define Nasion
outln = order_loop(slc.fwd_model.nodes(b,:),-1);
nasion= [-81.754 -8.157];
pp    = fourier_fit(outln(:,2:3),51);
% 0.457 seems to be a realistic place for inion, the other landmark
fr    = linspace(0,0.456,1001); fr(end) = [];
cap   = fourier_fit(pp,fr,nasion);
e_ar = [];
names = {};
[e_ar,names] = add_elec(e_ar,names, [0 cap(1,:) ],'Nz');
[e_ar,names] = add_elec(e_ar,names, [0 cap(100,:)],'Fpz');
[e_ar,names] = add_elec(e_ar,names, [0 cap(200,:)],'AFz');
[e_ar,names] = add_elec(e_ar,names, [0 cap(300,:)],'Fz');
[e_ar,names] = add_elec(e_ar,names, [0 cap(400,:)],'FCz');
[e_ar,names] = add_elec(e_ar,names, [0 cap(500,:)],'Cz');
[e_ar,names] = add_elec(e_ar,names, [0 cap(600,:)],'CPz');
[e_ar,names] = add_elec(e_ar,names, [0 cap(700,:)],'Pz');
[e_ar,names] = add_elec(e_ar,names, [0 cap(800,:)],'POz');
[e_ar,names] = add_elec(e_ar,names, [0 cap(900,:)],'Oz');
[e_ar,names] = add_elec(e_ar,names, [0 cap(end,:)],'Iz');

if show
   clf
   % show_fem(mdl);
   axis equal
   hold on
   slc.calc_colours.transparency_thresh = -1;
   slc.calc_colours.ref_level = 0.25;
   show_fem(slc);
end

% calculate a "horizontal plane between Nz and Iz
e1 = get_elec(e_ar,names,'Fpz');
e2 = get_elec(e_ar,names,'Oz');
d = e2 - e1;
t2 = -e1(2)/d(2);
t3 = -e1(3)/d(3);
z = e1(3) + t2*d(3);
y = e1(2) + t3*d(2);
slc = mdl_slice_mesher(mdl,[inf y z]);
slc.calc_colours.transparency_thresh = -1;
slc.calc_colours.ref_level = 0.25;
if show
   show_fem(slc);
   view(3)
end
slc.fwd_model.boundary = find_boundary(slc.fwd_model);
b = unique(slc.fwd_model.boundary);
outln = order_loop(slc.fwd_model.nodes(b,:),-1);
pp    = fourier_fit(outln(:,1:2),51);
fr = linspace(0,1,21); fr(end) = [];
pl = fourier_fit(pp,fr,e1(:,1:2));
pl(:,3) = e1(3) + d(3)*(pl(:,2) - e1(2))/d(2);
pl([1 11],:) = [];
nm = {'Fp1','AF7','F7','FT7','T7','TP7','P7','PO7','O1',...
      'O2','PO8','P8','TP8','T8','FT8','F8','AF8','Fp2'};
[e_ar names] = add_elecs(e_ar,names,pl,nm);


nm = {'AF3','AF4'};
[slc el] = define_plane(mdl,e_ar, names, 'AFz','AF7');
el = el([3 6],:);
[e_ar names] = add_elecs(e_ar,names,el,nm);
if show, show_fem(slc); end

nm = {'F9','F5','F3','F1','F2','F4','F6','F10'};
[slc el] = define_plane(mdl,e_ar, names, 'Fz','F7');
[e_ar names] = add_elecs(e_ar,names,el,nm);
if show, show_fem(slc); end

nm = {'FT9','FC5','FC3','FC1','FC2','FC4','FC6','FT10'};
[slc el] = define_plane(mdl,e_ar, names, 'FCz','FT7');
[e_ar names] = add_elecs(e_ar,names,el,nm);
if show, show_fem(slc); end

nm = {'T9','C5','C3','C1','C2','C4','C6','T10'};
[slc el] = define_plane(mdl,e_ar, names, 'Cz','T7');
[e_ar names] = add_elecs(e_ar,names,el,nm);
if show, show_fem(slc); end

nm = {'TP9','CP5','CP3','CP1','CP2','CP4','CP6','TP10'};
[slc el] = define_plane(mdl,e_ar, names, 'CPz','TP7');
[e_ar names] = add_elecs(e_ar,names,el,nm);
if show, show_fem(slc); end

nm = {'P9','P5','P3','P1','P2','P4','P6','P10'};
[slc el] = define_plane(mdl,e_ar, names, 'Pz','P7');
[e_ar names] = add_elecs(e_ar,names,el,nm);
if show, show_fem(slc); end

nm = {'PO3','PO4'};
[slc el] = define_plane(mdl,e_ar, names, 'POz','PO7');
el = el([3 6],:);
[e_ar names] = add_elecs(e_ar,names,el,nm);
if show, show_fem(slc); end
if show
   for i = 1:numel(names)
      plot_elec(e_ar,names,names{i});
   end
   hold off
end
% 
% load epos128;
% phi = sign(epos(:,1))*90 - epos(:,2); phi = pi*phi/180;
% the = epos(:,1); the(the>0) = -the(the>0); the = pi*the/180;
% z = sin(the);
% x = cos(the).*sin(phi);
% y = cos(the).*cos(phi);

function [e_ar names] = add_elecs(e_ar,names, e, name)
for i = 1:size(e,1)
    [e_ar names] = add_elec(e_ar,names,e(i,:),name{i});
end

function [e_ar names] = add_elec(e_ar,names,e,name);
e_ar(end+1,:) = e;
names{end+1} = name;
% function [x y z] = get_elec_cart(mdl,incl, azim)

function [e1] = get_elec(e_ar,names,name)
idx = find(ismember(names,name));
e1 = e_ar(idx,:);

function plot_elec(e_ar,names,name)
e = get_elec(e_ar,names,name);
if size(e,2) == 3
    plot3(e(:,1),e(:,2),e(:,3),'bo','MarkerFaceColor','b','MarkerSize',10);
else
    plot(e(:,1),e(:,2),'bo','MarkerFaceColor','b','MarkerSize',10);
end

function [slc, e] = define_plane(mdl, e_ar,names,n1, n2);
e1 = get_elec(e_ar,names,n1);
e2 = get_elec(e_ar,names,n2);
d = e2 - e1;
t2 = -e1(2)/d(2);
t3 = -e1(3)/d(3);
z = e1(3) + t2*d(3);
y = e1(2) + t3*d(2);
slc = mdl_slice_mesher(mdl,[inf y z]);
% show_fem(slc);
slc.calc_colours.transparency_thresh = -1;
slc.calc_colours.ref_level = 0.25;
slc.fwd_model.boundary = find_boundary(slc.fwd_model);

%exclude nasal cavity
b = unique(slc.fwd_model.boundary);
box = [  -16.0484  -19.0245
   22.6613  -15.7946
   23.7903  -92.9910
  -20.0806  -91.6990];
b = b(~inpolygon(slc.fwd_model.nodes(b,1),slc.fwd_model.nodes(b,3),box(:,1),box(:,2)));
outln = order_loop(slc.fwd_model.nodes(b,:),-1);
pp    = fourier_fit(outln(:,[1 3]),51);
fr = linspace(0,1,10001); fr(end) = [];
pl = fourier_fit(pp,fr,e1(:,[1 3]));
% find the position of the other electrode
di = sqrt(sum((pl - repmat(e2([1 3]),length(pl),1)).^2,2));
[jnk, p] = min(di);
p = p/10000;
if p > 0.5, p = 1-p; end
fr = [-5 -3 -2 -1 1 2 3 5] ./4 .* p;
el = fourier_fit(pp, fr, e1(:,[1 3]));
e(:,[1 3]) = el;
e(:,2) = e1(2) + d(2).*(e(:,3) - e1(3))./d(3);

% plot3(e(:,1),e(:,2),e(:,3),'o')
% plot3(e(1,1),e(1,2),e(1,3),'o','MarkerFaceColor','b')


%% UNIT TEST
function do_unit_test

warning('off','EIDORS:License');

% FMDL = MK_HEAD_MODEL_ADULT() 
% Returns a head fwd_model with no electrodes or tissues
fmdl = mk_head_model_adult();
unit_test_cmp('fwd_model', fmdl.type, 'fwd_model');
%
% FMDL = MK_HEAD_MODEL_ADULT(elec_pos, elec_shape)
% FMDL = MK_HEAD_MODEL_ADULT(elec_pos, elec_shape, maxh)
% Allows specifying electrode configuration, see PLACE_ELEC_ON_SURF
elec_pos = [8, 0, 0, 10]; % [N, equal angles, z1, z2];
elec_shape = [3, 0, 1]; % [radius, 0, maxh]
maxh = 10;
img1 = mk_head_model_adult(elec_pos, elec_shape); % default max is 8
img2 = mk_head_model_adult(elec_pos, elec_shape, maxh);
unit_test_cmp('custom + maxh', num_elems(img1) > num_elems(img2), true);

%
% FMDL = MK_HEAD_MODEL_ADULT(elecstr)
% A predefined model with round electrodes (radius 2 mm). 
%    '10-10'         - 73 labeled electrodes (10-10 placement system),
%                      no stimulation.
%    '2x16_planar'   - 2 rings of 16 electrodes, planar stimulation
%    '2x16_odd-even' - 2 rings, each stimulation is cross-plane 
%    '2x16_square'   - 2 rings, every second stimulation is cross-plane
%    '2x16_adjacent' - like square, but with adjacent stimulation
%    '1x32_ring'     - single ring of 32 electrodes
%    'no_elecs'      - no electrodes

img10 = mk_head_model_adult('10-10');
unit_test_cmp('10-10 # elecs', numel(img10.fwd_model.electrode),73);
img1 = mk_head_model_adult('1x32_ring');
unit_test_cmp('1x32 # elecs', numel(img1.fwd_model.electrode),32);
img1 = mk_head_model_adult('no_elecs');
unit_test_cmp('no elecs', isfield(img1.fwd_model, 'electrode'),false);
unit_test_cmp('materials', numel(img1.fwd_model.mat_idx), 4);

%
% IMG = MK_HEAD_MODEL_ADULT(... , tissuespec)
% Returns a head model containing separate regions defining scalp, skull, 
% CSF and brain as an image structure with conductivitity values. 
% TISSUESPEC must be one of:
%    'coarse'        - (defualt) tissue surfaces with maxh = 5mm
%    'fine'          - tissue surfaces with maxh = 2mm
%    'void'          - no internal tissues (void), useful for testing
%                      electrode positions

fmdl = mk_head_model_adult(elec_pos, elec_shape, maxh, 'void');
unit_test_cmp('void type', fmdl.type, 'fwd_model');
img1 = mk_head_model_adult('10-10', 'fine');
unit_test_cmp('fine', num_elems(img10) < num_elems(img1), true);

warning('on','EIDORS:License');