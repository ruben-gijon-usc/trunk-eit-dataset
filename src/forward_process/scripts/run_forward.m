% run_forward.m — Run the EIDORS forward simulation given a conductivity grid.
%
% This is the main entry-point called by the Python forward_process layer.
% It orchestrates:
%   1. EIDORS initialisation
%   2. FEM model creation          (create_model)
%   3. Grid→element conductivity   (assign_conductivity)
%   4. FEM forward solve           (fwd_solve)
%   5. Writing results to tmpdir
%
% Usage (from Octave CLI or Python subprocess):
%   run_forward(n_electrodes, grid_path, x_min, x_max, y_min, y_max, fallback_cond, out_dir)
%
% Inputs:
%   n_electrodes   — number of surface electrodes
%   grid_path      — path to conductivity grid text file (produced by Python)
%   x_min, x_max   — physical x extent of the domain (metres)
%   y_min, y_max   — physical y extent of the domain (metres)
%   fallback_cond  — conductivity for out-of-domain elements
%   out_dir        — directory where output files are written
%
% Output files written to out_dir:
%   nodes.txt     — FEM node coordinates   [N × 2]
%   elems.txt     — FEM connectivity       [E × 3], 1-indexed
%   elem_data.txt — element conductivities [E × 1]
%   voltages.txt  — boundary voltage measurements [M × 1]

function run_forward(n_electrodes, grid_path, x_min, x_max, y_min, y_max, fallback_cond, out_dir)
    % --- 1. Initialise EIDORS ------------------------------------------------
    % mfilename('fullpath') is reliable even when called via addpath from a driver
    % NOTE: addpath is already set by the Python driver before calling this func.
    startup_eidors();

    % --- 2. Build FEM model --------------------------------------------------
    [~, img] = create_model(n_electrodes);

    % --- 3. Assign conductivity from grid ------------------------------------
    img = assign_conductivity(img, grid_path, x_min, x_max, y_min, y_max, fallback_cond);

    % --- 4. Forward solve ----------------------------------------------------
    vh = fwd_solve(img);

    if ~isfield(vh, 'meas')
        error('run_forward: forward solve did not produce boundary measurements (vh.meas missing).');
    end

    % --- 5. Write outputs ----------------------------------------------------
    dlmwrite(fullfile(out_dir, 'nodes.txt'),     img.fwd_model.nodes, ' ');
    dlmwrite(fullfile(out_dir, 'elems.txt'),     img.fwd_model.elems, ' ');
    dlmwrite(fullfile(out_dir, 'elem_data.txt'), img.elem_data,        ' ');
    dlmwrite(fullfile(out_dir, 'voltages.txt'),  vh.meas,              ' ');
end
