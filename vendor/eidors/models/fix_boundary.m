function mdl = fix_boundary(mdl)
%FIX_BOUNDARY Orient boundary triangles consistently
% mdl = FIX_BOUNDARY(mdl) changes or adds mdl.boundary to an eidors
%  fwd_model
% 
% See also FIX_MODEL

% (C) 2015-2016 Bartlomiej Grychtol. License: GPL version 2 or 3
% $Id: fix_boundary.m 6926 2024-05-31 15:34:13Z bgrychtol $

opt.elem2face  = 1;
opt.boundary_face = 1;
opt.inner_normal = 1;
tmp = fix_model(mdl,opt);
flip = tmp.elem2face(logical(tmp.boundary_face(tmp.elem2face).*tmp.inner_normal));
tmp.faces(flip,:) = tmp.faces(flip,[1 3 2]);
tmp.normals(flip,:) = -tmp.normals(flip,:);

% don't pollute the model, but make existing relevant fields consistent
fields = {'faces', 'normals', 'inner_normal', 'boundary_face'};
for i = 1:numel(fields)
   if isfield(mdl, fields{i})
      mdl.(fields{i}) = tmp.(fields{i}); 
   end
end
mdl.boundary = tmp.faces(tmp.boundary_face,:);

mdl = eidors_obj('set', mdl);
