function img = set_slice(img, level, spec)
% SET_SLICE  Set parameters for slice display.
%   IMG = SET_SLICE(IMG, LEVEL, SPEC) sets the parameters on the image
%   struct IMG the parameters required by MDL_SLICE_MAPPER.
%
%   PARAMETERS:
%       IMG     - an eidors image or fwd_model
%       LEVEL   - any definition accepted by LEVEL_MODEL_SLICE for 
%                 a single slice, or [] to leave current specification
%                 unchanged.
%       SPEC    - specification of interpolations points, in one of the
%                 following forms
%           R           - scalar resolution (points per unit)
%                         Sets mdl_slice_mapper.resolution.
%           [npx, npy]  - 2-element vector specifying horizontal and
%                         vertical number of points
%                         Sets mdl_slice_mapper.npx/npy.
%           [np, 0]
%           [0, np]     - 2-element vector, one element is zero. 
%                         Sets img.mdl_slice_mapper.npoints,
%                         the number of points across the larger.
%           {xpts,ypts} - 2-elements cell array containing vectors of
%                         points in horizontal and vertical directsion.
%                         Sets mdl_slice_mapper.x_pts/y_pts.
%
%   SET_SLICE will remove any alternative specifications of point locations
%   from the mdl_slice_mapper field (fields npx/npy, xpts/ypts, resolution,
%   npoints). If LEVEL is [], mdl_slice_mapper.level is not changed if
%   present, and not added if absent. Otherwise mdl_slice_mapper.level will
%   be replaced, and mdl_slice_mapper.center and .rotate (deprecated) will
%   be removed.
%
%   SET_SLICE does not modify the mdl_slice_mapper.model_2d field.
%
%   SET_SLICE sets img.show_slices.axes_msm = true. See SHOW_SLICES.
%
% Example:
%       img = set_slice(img, LEVEL, [npx, npy]);
%   is equivalent to:
%       img.fwd_model.mdl_slice_mapper.level = LEVEL;
%       img.fwd_model.mdl_slice_mapper.npx = npx;
%       img.fwd_model.mdl_slice_mapper.npy = npy;
%
% See also MDL_SLICE_MAPPER, LEVEL_MODEL_SLICE, SHOW_SLICES

% (C) 2024 Bartek Grychtol. 
% License: GPL version 2 or version 3
% $Id: set_slice.m 6799 2024-04-20 19:44:10Z aadler $

if ischar(img) && strcmp(img,'UNIT_TEST'); do_unit_test; return; end

switch img.type
    case 'image'
        fmdl = img.fwd_model;
        isimage = true;
    case 'fwd_model'
        fmdl = img;
        isimage = false;
    otherwise
        error('EIDORS:WrongInput', 'Expected eidors image or fwd_model.')
end

fmdl = do_set_slice(fmdl, level, spec);

if isimage
    img.fwd_model = fmdl;
    img.show_slices.axes_msm = true;
else
    img = fmdl;
end


function fmdl = do_set_slice(fmdl, level, spec)

if ~isempty(level)
    fmdl.mdl_slice_mapper.level = level;
    try fmdl.mdl_slice_mapper = rmfield(fmdl.mdl_slice_mapper, {'centre', 'rotate'}); end
end

if ~isempty(spec)
    try fmdl.mdl_slice_mapper = rmfield(fmdl.mdl_slice_mapper, {'x_pts', 'y_pts'}); end
    try fmdl.mdl_slice_mapper = rmfield(fmdl.mdl_slice_mapper, {'npx', 'npy'}); end
    try fmdl.mdl_slice_mapper = rmfield(fmdl.mdl_slice_mapper, 'npoints'); end
    try fmdl.mdl_slice_mapper = rmfield(fmdl.mdl_slice_mapper, 'resolution'); end
end

if iscell(spec) % {xpts, ypts}
    fmdl.mdl_slice_mapper.x_pts = spec{1};
    fmdl.mdl_slice_mapper.y_pts = spec{2};
elseif isnumeric(spec)
    switch numel(spec)
        case 1 % resolution
            fmdl.mdl_slice_mapper.resolution = spec;
        case 2 % [npx, npy] or [np, 0]
            if any(spec == 0)
                fmdl.mdl_slice_mapper.npoints = max(spec);
            else
                fmdl.mdl_slice_mapper.npx = spec(1);
                fmdl.mdl_slice_mapper.npy = spec(2);
            end
        otherwise
            error('EIDORS:WrongInput', 'Expected 1- or 2- element SPEC');
            
    end      
else
    error('EIDORS:WrongInput','Point definition SPEC not understood.');
end



function do_unit_test()
    figure(1)
    show_slices('UNIT_TEST');
    figure(2)
    do_unit_test_show_slices()
    figure(3);
    mdl_slice_mapper('UNIT_TEST');
    figure(4);
    do_unit_test_mdl_slice_mapper()
    

function do_unit_test_show_slices()
   clf; sp=0;
   %1
   img=calc_jacobian_bkgnd(mk_common_model('a2t3',8)); 
   img.elem_data=toeplitz(1:size(img.fwd_model.elems,1),1);
   sp=sp+1;subplot(4,5,sp); show_slices(img) 
   %2
   %img.calc_colours.npoints= 128;
   img = set_slice(img, [], [128, 0]);
   sp=sp+1;subplot(4,5,sp); show_slices(img) 
   %3
%    img.calc_colours.npoints= 32;
   img = set_slice(img, [], [0 32]);
   img.elem_data=toeplitz(1:size(img.fwd_model.elems,1),1:3);
   sp=sp+1;subplot(4,5,sp); show_slices(img) 

   %4
   img.show_slices.img_cols= 1;
   sp=sp+1;subplot(4,5,sp); show_slices(img) 

   %5
   imgn = rmfield(img,'elem_data');
   imgn.node_data=toeplitz(1:size(img.fwd_model.nodes,1),1);

   img.elem_data = img.elem_data(:,1);
%    img.fwd_model.mdl_slice_mapper.npx = 10;
%    img.fwd_model.mdl_slice_mapper.npy = 20;
%    img.fwd_model.mdl_slice_mapper.level = [inf,inf,0];
   img = set_slice(img, [inf,inf,0], [10, 20]);
   sp=sp+1;subplot(4,5,sp); show_slices(img);

   %6
   img.elem_data = img.elem_data(:,1);
%    img.fwd_model.mdl_slice_mapper = rmfield(img.fwd_model.mdl_slice_mapper, {'npx','npy'});
%    img.fwd_model.mdl_slice_mapper.x_pts = linspace(-100,100,20);
%    img.fwd_model.mdl_slice_mapper.y_pts = linspace(-150,150,30);
%    img.fwd_model.mdl_slice_mapper.level = [inf,inf,0];
   img = set_slice(img,[inf,inf,0], {linspace(-100,100,20),linspace(-150,150,30)}); 
   sp=sp+1;subplot(4,5,sp); show_slices(img);

   %7
   sp=sp+1;subplot(4,5,sp); show_slices(imgn) 

   %8
%    imgn.fwd_model.mdl_slice_mapper.x_pts = linspace(-100,100,20);
%    imgn.fwd_model.mdl_slice_mapper.y_pts = linspace(-150,150,30);
%    imgn.fwd_model.mdl_slice_mapper.level = [inf,inf,0];
   imgn = set_slice(imgn,[inf,inf,0], {linspace(-100,100,20),linspace(-150,150,30)}); 
   sp=sp+1;subplot(4,5,sp); show_slices(imgn) 


% 3D images
   %9
   img=calc_jacobian_bkgnd(mk_common_model('n3r2',[16,2])); 
%    img.calc_colours.npoints= 16;
   img = set_slice(img, [], [0 16]);
   img.elem_data=toeplitz(1:size(img.fwd_model.elems,1),1);
   sp=sp+1;subplot(4,5,sp); show_slices(img,2) 

   %10
   img.elem_data=img.elem_data*[1:3];
   sp=sp+1;subplot(4,5,sp); show_slices(img,2) 

   %11
   img.elem_data=img.elem_data(:,1:2);
   sp=sp+1;subplot(4,5,sp); show_slices(img,[inf,inf,1;0,inf,inf;0,1,inf]);

   %12
   img.show_slices.sep = 5;
%    img.fwd_model.mdl_slice_mapper.x_pts = linspace(-1,1,20);
%    img.fwd_model.mdl_slice_mapper.y_pts = linspace(-1,1,30);
%    img.fwd_model.mdl_slice_mapper.level = [inf,inf,0];
   img = set_slice(img, [inf,inf,0], {linspace(-1,1,20), linspace(-1,1,30)});
   sp=sp+1;subplot(4,5,sp); show_slices(img,2) 

   % tests 13--19 of calc_slices are not relevant

function do_unit_test_mdl_slice_mapper()
% 2D NUMBER OF POINTS
   imdl = mk_common_model('a2c2',8); fmdl = imdl.fwd_model;
   fmdl.nodes = 1e-15 * round(1e15*fmdl.nodes);
%    fmdl.mdl_slice_mapper.level = [inf,inf,0];
%    fmdl.mdl_slice_mapper.npx = 5;
%    fmdl.mdl_slice_mapper.npy = 5;
   fmdl = set_slice(fmdl, [inf,inf,0], [5, 5]);
   eptr = mdl_slice_mapper(fmdl,'elem');
   
   si = @(x,y) sub2ind([5,5],x, y);
   % points lie on nodes and edges, we allow any associated element
   els = {si(2,2), [25,34];
          si(2,4), [27,30];
          si(3,3), [1,2,3,4];
          si(4,2), [33,36];
          si(4,4), [28,31];
          };
   
   tst = eptr == [    0   37   38   39    0
                     46   25    5   27   42
                     47    8    1    6   41
                     48   33    7   28   40
                      0   45   44   43    0];
                      
   for i = 1:size(els,1)
    tst(els{i,1}) = ismember(eptr(els{i,1}), els{i,2});
   end
   unit_test_cmp('eptr01',tst,true(5,5));

   nptr = mdl_slice_mapper(fmdl,'node');
   test = [   0   27   28   29    0
             41    6    7    8   31
             40   13    1    9   32
             39   12   11   10   33
              0   37   36   35    0];
   unit_test_cmp('nptr01',nptr,test);

   nint = mdl_slice_mapper(fmdl,'nodeinterp');
   unit_test_cmp('nint01a',nint(2:4,2:4,1),[ 0.2627, 0.6909, 0.2627; 
                                             0.6906, 1,      0.6906;
                                             0.2627, 0.6909, 0.2627], 1e-3);

%    fmdl.mdl_slice_mapper.npx = 6;
%    fmdl.mdl_slice_mapper.npy = 4;
   fmdl = set_slice(fmdl, [], [6 4]);
   eptr = mdl_slice_mapper(fmdl,'elem');
   res = [  0   49   38   38   52    0
           62   23    9   10   20   55
           63   24   14   13   19   54
            0   60   44   44   57    0];
   unit_test_cmp('eptr02',eptr,res);

   nptr = mdl_slice_mapper(fmdl,'node');
   res = [  0   27   15   16   29    0
           25    6    2    3    8   18
           24   12    5    4   10   19
            0   37   22   21   35    0];
   unit_test_cmp('nptr02',nptr,res);

% DIRECT POINT TESTS
   imdl = mk_common_model('a2c2',8); fmdl = imdl.fwd_model;
%    fmdl.mdl_slice_mapper.level = [inf,inf,0];
%    fmdl.mdl_slice_mapper.x_pts = linspace(-.95,.95,4);
%    fmdl.mdl_slice_mapper.y_pts = [0,0.5];
   fmdl = set_slice(fmdl, [inf,inf,0], {linspace(-.95,.95,4), [0,0.5]});
   eptr = mdl_slice_mapper(fmdl,'elem');
   unit_test_cmp('eptr03',eptr,[ 47 8 6 41; 0 25 27 0]);

   nptr = mdl_slice_mapper(fmdl,'node');
   unit_test_cmp('nptr03',nptr,[ 40 13 9 32; 0 6 8 0]);

% 3D NPOINTS
   imdl = mk_common_model('n3r2',[16,2]); fmdl = imdl.fwd_model;
%    fmdl.mdl_slice_mapper.level = [inf,inf,1]; % co-planar with faces
%    fmdl.mdl_slice_mapper.npx = 4;
%    fmdl.mdl_slice_mapper.npy = 4;
   fmdl = set_slice(fmdl,[inf,inf,1],[4, 4]);
   eptr = mdl_slice_mapper(fmdl,'elem');
   si = @(x,y) sub2ind([4,4],x, y);
   % points lie on nodes and edges, we allow any associated element
   els = {si(1,2), [ 42,317];
          si(1,3), [ 30,305];
          si(2,1), [ 66,341];
          si(2,2), [243,518];
          si(2,3), [231,506];
          si(2,4), [  6,281];
          si(3,1), [ 78,353];
          si(3,2), [252,527];
          si(3,3), [264,539];
          si(3,4), [138,413];
          si(4,2), [102,377];
          si(4,3), [114,389];};
   tst = eptr == zeros(4); 
   for i = 1:size(els,1)
    tst(els{i,1}) = ismember(eptr(els{i,1}), els{i,2});
   end
   unit_test_cmp('eptr04',tst, true(4));
   nptr = mdl_slice_mapper(fmdl,'node');
   test = [   0   101    99     0
            103   116   113    97
            105   118   121   111
              0   107   109     0];
   unit_test_cmp('nptr04',nptr, test);

   fmdl.mdl_slice_mapper.level = [inf,0.01,inf];
   eptr = mdl_slice_mapper(fmdl,'elem'); 
   test = [621   792   777   555
           345   516   501   279
           343   515   499   277
            69   240   225     3];
   unit_test_cmp('eptr05',eptr,test);

   nptr = mdl_slice_mapper(fmdl,'node');
   test = [230   250   248   222
           167   187   185   159
           104   124   122    96
            41    61    59    33];
   unit_test_cmp('nptr05',nptr,test);

   nint = mdl_slice_mapper(fmdl,'nodeinterp');
   unit_test_cmp('nint05a',nint(2:3,2:3,4),[.1250,.1250;.8225,.8225],1e-3);
   
% Centre and Rotate
   fmdl.mdl_slice_mapper = rmfield(fmdl.mdl_slice_mapper,'level');
   fmdl.mdl_slice_mapper.centre = [0,0,0.9];
   fmdl.mdl_slice_mapper.rotate = eye(3);
%    fmdl.mdl_slice_mapper.npx = 4;
%    fmdl.mdl_slice_mapper.npy = 4;
   fmdl = set_slice(fmdl, [], [4,4]);
   eptr = mdl_slice_mapper(fmdl,'elem');
   test = [  0    42    30     0
            66   243   229     6
            78   250   264   138
             0   102   114     0];
   unit_test_cmp('eptr06',eptr, test);

% SLOW
   imdl = mk_common_model('d3cr',[16,3]); fmdl = imdl.fwd_model;
   fmdl.nodes = 1e-15*round(1e15*fmdl.nodes);
%    fmdl.mdl_slice_mapper.level = [inf,inf,1];
%    fmdl.mdl_slice_mapper.npx = 64;
%    fmdl.mdl_slice_mapper.npy = 64;
   fmdl = set_slice(fmdl, [inf,inf,1], [64, 64]);
   t = cputime;
   eptr = mdl_slice_mapper(fmdl,'elem');
   txt = sprintf('eptr10 (t=%5.3fs)',cputime - t);
% Note that triangulation gives different
% results near the edge
   test = [122872   122872   122748
           122809   122749   122689
           122749   122749   122689];
   unit_test_cmp(txt,eptr(10:12,11:13),test);  
   

% CHECK ORIENTATION
   imdl=mk_common_model('a2c0',16);
   img= mk_image(imdl,1); img.elem_data(26)=1.2;
   subplot(231);show_fem(img);
   subplot(232);show_slices(img);
%    img.fwd_model.mdl_slice_mapper.npx= 20;
%    img.fwd_model.mdl_slice_mapper.npy= 30;
%    img.fwd_model.mdl_slice_mapper.level= [inf,inf,0];
   img = set_slice(img, [inf,inf,0], [20, 30]);
   
% Remaining tests in mdl_slice_mapper are not relevant to set_slice