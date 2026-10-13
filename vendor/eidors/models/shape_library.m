function out = shape_library(action, shape, varargin)
%SHAPE_LIBRARY Common shapes for models
%  SHAPE_LIBRARY('list') lists available shapes (strings)
%  
%  SHAPE_LIBRARY('list',SHAPE) lists available outlines for the given shape
%  
%  SHAPE_LIBRARY('show', SHAPE) plots available outlines for the given shape
%  overlayed on the image from which they were segmented
%
%  SHAPE_LIBRARY('get',SHAPE) returns a struct with all the information for 
%  the given shape, including at least the fields:
%     .boundary  - a set of [x y] points defining the boundary
%     .pic.img   - an image from which the contours were extracted
%     .pic.X 
%     .pic.Y     - coordinate vectors to use with IMAGESC
%     .copyright - source, authors and license
%  Additional outlines of internal objects (e.g. lungs) are defined in
%  separate fields akin to .boundary above.
%
%  SHAPE_LIBRARY('get', SHAPE, FIELD [,FIELD2,..]) 
%  SHAPE_LIBRARY('get', SHAPE, {FIELD [,FIELD2,...]})
%  returns a requested outline FIELD from SHAPE. If more than one FIELD is 
%  requested, a cell array of [x y] matries is returned
% 
%  SHAPE_LIBRARY('add') displays information on adding new shapes
%
%EXAMPLES:
% shape_library('list');
% shape_library('list','pig_23kg');
% shape_library('get','pig_23kg','boundary','lungs')

% (C) 2011 Bartlomiej Grychtol. License: GPL version 2 or version 3
% $Id: shape_library.m 6626 2023-11-21 15:35:02Z aadler $

n_IN = nargin;
if n_IN < 2
    shape = '';
end

switch action
    case 'UNIT_TEST'
        do_unit_test;
        return
    case 'list'
        out = list_shapes(shape);
    case 'show'
        show_shape(shape);
    case 'get'
        out = get_shape(shape,varargin);
    case 'add'
	add_shape;
end

function files = get_shape_list()
    % shapes are stored in dir shape_library/
    pth = mfilename('fullpath');
    file_list = dir([pth,'/*.mat']);
    if isempty(file_list)
        error('Can''t find shape_library');
    end
    for i=1:length(file_list)
        % remove ".mat"
        files{i,1} = file_list(i).name(1:end-4);
    end

function out = list_shapes(shape)
    out = get_shape_list();
if isempty(shape)
    disp('Available shapes:');
else
%   s = load('shape_library.mat',shape);
    s = cell2struct(out,out);
    if ~isfield(s,shape)
       not_found(shape); out=[]; return;
    else
       out = get_shape(shape,'');
    end
    disp(['Shape ' shape ' contains:']);     
end
disp(out);

function show_shape(shape)

if isempty(shape)
    eidors_msg('SHAPE_LIBRARY: you must specify a shape to show');
    return
end
s = get_shape(shape,'');
%figure 
colormap gray
imagesc(s.pic.X, s.pic.Y, s.pic.img);
set(gca,'YDir','normal');
hold all
str = {};

fields = fieldnames(s);
for i = 1:numel(fields)
    if strcmp(fields{i},'electrodes'), continue, end
    if isnumeric(s.(fields{i})) && size(s.(fields{i}),2)==2
        plot(s.(fields{i})(:,1),s.(fields{i})(:,2),'-o','LineWidth',2)
	str =[str fields(i)];
    end
end
if isfield(s,'electrodes')
   el = s.electrodes;
   h = plot(el(:,1),el(:,2),'*');
   c = get(h, 'Color');
   str = [str {'electrodes'}];
   for i = 1:length(el)
   	text(1.1*el(i,1),1.1*el(i,2),num2str(i),...
	   'HorizontalAlignment','Center','Color',c);
   end
end
legend(str,'Location','NorthEastOutside','Interpreter','none');
axis equal
axis tight
title(shape,'Interpreter','none')
xlabel(s.copyright)
hold off



function out = get_shape(shape,fields)
if isempty(shape)
    eidors_msg('SHAPE_LIBRARY: you must specify a shape to get');
    out=[];
    return
end

try
  out=load([mfilename('fullpath'),'/',shape]);
catch
  not_found(shape); out=[]; return;
end

if ~isempty(fields)
    if iscell(fields{1}) fields = fields{1}; end
    s = out; out = {};
    for i = 1:numel(fields)
	out(i) = {s.(fields{i})};
    end
    if i == 1, out = out{1}; end
end


function add_shape
eidors_msg(['SHAPE_LIBRARY: To contribute a shape contact ' ...
	'<a href="http://eidors3d.sourceforge.net/faq.shtml#maintainers">' ...
 	'EIDORS maintainers</a>']);


function not_found(shape)
eidors_msg(['SHAPE_LIBRARY: Didn''t find shape ' shape]);


function do_unit_test
    shape_library('list');
    shape_library('list','a-shape-we-dont-have');% give error
    shape_library('list','pig_23kg');
    shape_library('show'); % fail gracefully
    shape_library('show','pig_23kg');
    shape_library('get'); %fail gracefully
    shape_library('get','a-shape-we-dont-have');% give error
    out=shape_library('get','pig_23kg');% give a struct
    out=shape_library('get','pig_23kg','heart');% give array
    out=shape_library('get','pig_23kg','boundary','lungs');% give cell array
    out=shape_library('get','pig_23kg',{'boundary','lungs'});% give cell array
    shape_library('add');
