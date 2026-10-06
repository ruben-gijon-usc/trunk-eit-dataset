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

function [fmdl, img] = create_model(n_electrodes, pattern_str)
    if nargin < 1
        n_electrodes = 16;
    end
    if nargin < 2
        pattern_str = 'adjacent';
    end

    % Adjust mesh density based on the number of electrodes to match the paper's detail level
    if n_electrodes >= 32
        model_name = 'd2d3c';
    else
        model_name = 'd2d2c';
    end

    imdl = mk_common_model(model_name, n_electrodes);
    fmdl = imdl.fwd_model;

    % Override stimulation pattern (0.010 A as in the paper)
    if strcmp(pattern_str, 'adjacent')
        [stim, ~] = mk_stim_patterns(n_electrodes, 1, '{ad}', '{ad}', {}, 0.010);
    elseif strcmp(pattern_str, 'opposite')
        [stim, ~] = mk_stim_patterns(n_electrodes, 1, '{op}', '{op}', {}, 0.010);
    else
        [stim, ~] = mk_stim_patterns(n_electrodes, 1, '{ad}', '{ad}', {}, 0.010);
    end

    fmdl.stimulation = stim;
    img  = mk_image(fmdl, 1.0);
end
