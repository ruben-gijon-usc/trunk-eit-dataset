function write_ibex_mat(fname, fmdl, imgs, frame_rate)
% WRITE_IBEX_MAT:  write images to ibex mat file format
%  write_ibex_mat(fname, fmdl, imgr)
%    fname - file name to write to 
%    fmdl  - the fmdl to use, including lung regions
%    imgr  - the reconstructed image requence
%    frame_rate - frame rate of acq data
%
%
% Limitations, imgr must be a 2D 32×32 image

% (C) 2024 Andy Adler. License: GPL version 2 or version 3
% $Id: write_ibex_mat.m 7002 2024-11-24 13:11:35Z aadler $

[llROI, rlROI, thROI] = get_ROIs(fmdl);
imgs(isnan(imgs))= 0;
data = create_data(imgs, frame_rate, thROI, llROI, rlROI );
save(fname,'data');

function [llung_ROI, rlung_ROI, thorax_ROI] = get_ROIs(fmdl);
    lmi = length(fmdl.mat_idx);
    img = mk_image(fmdl, 1); % background conductivity
    if lmi==1;
      error('Need mat_idx regions to segment lungs');
    elseif lmi==2; % one lung, must cut
      img.elem_data(fmdl.mat_idx{2}) = 0.3; % lungs
    else % at least two lungs
      img.elem_data(fmdl.mat_idx{2}) = 0.3; % lungs
      img.elem_data(fmdl.mat_idx{3}) = 0.3; % lungs
    end
    img.calc_colours.npoints = 32;
    ROI = calc_slices(img,1);
    if ~(size(ROI)==[32,32])
       error('image size must be 32×32'); 
    end
     
    thorax_ROI= ~isnan(ROI);
    lung_ROI  = thorax_ROI & (ROI==0.3);
    xpts = linspace(-1,1,32);
    rlung_ROI = lung_ROI & (xpts<0);
    llung_ROI = lung_ROI & (xpts>0);

function data = create_data(imgs, FR, thorax_ROI, llung_ROI, rlung_ROI);
    data.imageRate = FR;

    data.patient.ROI.Inside =thorax_ROI*100; % to scale it up to 100
    data.patient.ROI.RightLung =rlung_ROI*100;
    data.patient.ROI.LeftLung =llung_ROI*100;
    data.patient.ROI.Heart =zeros(size(imgs,1),size(imgs,2));

    % put to dummy because they are missing
    data.patient.halfChest = 'NaN';
    data.patient.height = 'NaN';
    data.patient.weight = 'NaN';
    data.patient.gender = 'NaN';

    data.measurement.Position.transversal = zeros (1,size(imgs,3));
    data.measurement.Position.longitudinal = zeros (1,size(imgs,3));
    data.measurement.ImageQuality = 100*ones(1,size(imgs,3));
    data.measurement.ElectrodeQuality = zeros(size(imgs,3),32);
    data.measurement.ZeroRef = imgs;

    data.injctionPattern= 'NaN';
    data.SensorBelt.NumEl= 'NaN';

    data.measurement.CompositValue=squeeze(sum(sum(imgs,2),1));


