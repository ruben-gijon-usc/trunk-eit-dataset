% Create circular EIT model with electrodes
% Input: n_electrodes (default 16)
% Output: fwd_model, img (image with homogeneous conductivity)

function [fmdl, img] = create_model(n_electrodes)
    if nargin < 1
        n_electrodes = 16;
    end

    run('/home/ruben/Documentos/EIDORS_implementation/eidors_lib/eidors-v3.12/eidors/startup.m');

    imdl = mk_common_model('d2d1c', n_electrodes);
    img = mk_image(imdl, 1.0);
    fmdl = img.fwd_model;
end