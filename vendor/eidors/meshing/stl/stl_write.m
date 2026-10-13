function stl_write(fv, name, type)
%STL_WRITE Create an STL file from a patch struct
%
% STL_WRITE(mdl, name) where:
%  mdl is an eidors fwd_model structure. 
%  name is the file name (character string), if no extension given, 'stl'
%  assumed.
%  STL_WRITE will use mdl.elems if they are triangles. Otherwise, it will
%  use mdl.boundary or, if absent, call find_boundary.
%
% STL_WRITE(fv, name) where:
%  fv is face-vertex (patch) structure (fv.faces, fv.vertices)
%  name is the file name (character string), if no extension given, 'stl'
%  assumed.
%
% STL_WRITE(fv, name, type) specifies the type of file to write:
%  'bin'  - binary file (default)
%  'txt'  - ASCII file
%
% See also: STL_READ, FIND_BOUNDARY

% (C) 2006 Eric Carlson: Public Domain
% Adapted by Bartlomiej Grychtol from:
% http://www.mathworks.com/matlabcentral/newsreader/view_thread/126215
% $Id: stl_write.m 6836 2024-05-02 08:24:41Z bgrychtol $


[folder, label, ext] = fileparts(name);
if isempty(ext), ext = '.stl'; end
if isempty(folder)
   name = [label ext];
else
   name = [folder filesep label ext];
end

if isfield(fv, 'type') && strcmp(fv.type, 'fwd_model')
   fv.vertices = fv.nodes;
   if size(fv.elems,2)== 3
       fv.faces = fv.elems;
   else
       try
           fv.faces = fv.boundary;
       catch
           fv.faces = find_boundary(fv);
       end
   end
end
fv.vertices = double(fv.vertices);
v1 = fv.vertices(fv.faces(:,2),:)-fv.vertices(fv.faces(:,1),:);
v2 = fv.vertices(fv.faces(:,3),:)-fv.vertices(fv.faces(:,2),:);
Norms = cross3(v1,v2);
clear v1 v2
fv.vertices = single(fv.vertices);
v1(:,1:3) = fv.vertices(fv.faces(:,1),1:3);
v2(:,1:3) = fv.vertices(fv.faces(:,2),1:3);
v3(:,1:3) = fv.vertices(fv.faces(:,3),1:3);

if nargin < 3 
    type = 'bin';
end
switch type
    case 'bin'
        write_bin_stl(name, label, Norms, v1, v2, v3);
    case 'txt' 
        write_txt_stl(name, label, Norms, v1, v2, v3);
    otherwise
        error('EIDORS:WrongInput','Type can only be ''bin'' or ''txt''');
end


function write_bin_stl(name, label, Norms, v1, v2, v3)
progress_msg('Writing binary STL file...')
fid = fopen(name,'wb','ieee-le');
header = label;
if length(header) > 80
    header = header(1:80);
else
    header(end+1:80) = '-';
end
fwrite(fid, header, 'char');
nf = size(v1,1);
fwrite(fid, uint32(nf), 'uint32');
for k = 1:nf
    if mod(k,100)==0, progress_msg(k/nf); end
    fwrite(fid, [Norms(k,:), v1(k,:), v2(k,:), v3(k,:)], 'float32');
    fwrite(fid, 0,'uint16');
end
fclose(fid);
progress_msg(Inf)


function write_txt_stl(name, label, Norms, v1, v2, v3)
progress_msg('Writing ASCII STL file...')
fid = fopen(name,'w');
fprintf(fid,'solid %s\n',label);
nf = size(v1,1);
for k = 1:nf
    if mod(k,100)==0, progress_msg(k/nf); end
    fprintf(fid,['facet normal %.12g %.12g %.12g\n' ... 
                 '\t' 'outer loop\n' ... 
                 '\t\t' 'vertex %.12g %.12g %.12g\n' ...
                 '\t\t' 'vertex %.12g %.12g %.12g\n' ...
                 '\t\t' 'vertex %.12g %.12g %.12g\n' ...
                 '\t' 'endloop\n' ...
                 'endfacet\n'], ...
        Norms(k,1),Norms(k,2),Norms(k,3), ...
        v1(k,1), v1(k,2), v1(k,3), ...
        v2(k,1), v2(k,2), v2(k,3), ...
        v3(k,1), v3(k,2), v3(k,3) );
end
fprintf(fid,'endsolid %s\n',label);
fclose(fid);
progress_msg(Inf)

function M=cross3(r,F)
% function to calculate normalized cross product rxF/|rxF|
% handles (same-size) arrays (n by 3) for r and F
%
       M = [(r(:,2).*F(:,3) - r(:,3).*F(:,2)) ...
            (r(:,3).*F(:,1) - r(:,1).*F(:,3)) ...
            (r(:,1).*F(:,2) - r(:,2).*F(:,1))];
       M_mag = sqrt(sum((M.*M)')');
       M(:,1) = M(:,1)./M_mag;
       M(:,2) = M(:,2)./M_mag;
       M(:,3) = M(:,3)./M_mag;
