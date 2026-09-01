% create_model.m — Build circular EIDORS FEM model with equi-spaced electrodes.
%
% Usage:
%   [fmdl, img] = create_model(n_electrodes)
%
% Inputs:
%   n_electrodes  — number of surface electrodes (default: 16)
%
% Outputs:
%   fmdl  — forward model struct
%   img   — EIDORS image with uniform conductivity = 1.0 S/m

function [fmdl, img] = create_model(n_electrodes)
    if nargin < 1
        n_electrodes = 16;
    end

    imdl = mk_common_model('d2d1c', n_electrodes);
    img  = mk_image(imdl, 1.0);
    fmdl = img.fwd_model;
end
