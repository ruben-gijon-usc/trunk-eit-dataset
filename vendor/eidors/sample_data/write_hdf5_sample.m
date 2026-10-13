% Write HDF5 Sample data
% Possner, Bulst, Adler "HDF5-based data format for EIT data", Conf. EIT2023
% $Id: write_hdf5_sample.m 7124 2024-12-29 15:18:32Z aadler $
function fname = write_and_test;
  fname = 'montreal_data_1995_breathold.h5';
  delete(fname)
  hdf5_sample_write
  [dd,~,stim] = eidors_readdata('montreal_data_1995_breathold.h5','h5','datasetEIT');
  imdl = mk_common_model('c2t2',16);
  imdl.fwd_model = rmfield(imdl.fwd_model,'meas_select');
  imdl.fwd_model.stimulation = stim;
  imgr = inv_solve(imdl, dd(:,1), dd);
  show_slices(imgr);

function hdf5_sample_write
   load 'montreal_data_1995.mat'
   txt = fileread('montreal_data_1995.readme');
   h5createwrite('/VERSION',2023.4,'double');
   h5createwrite('/patient/Readme',txt,'string');
   [stim,msel] = mk_stim_patterns(16,1,[0,1],[0,1],{'no_meas_current'},1);
   freq=13000; oo=ones(1,13); FR = 1/7; % 13 meas / frame
   stimprot= []; measprot= [];
   for i=1:length(stim);
      stimprot = [stimprot, stim(i).stim_pattern*oo];
      measprot = [measprot, stim(i).meas_pattern'];
   end
   h5createwrite('/data/datasetEIT/Meas.V.Abs',zc_breathhold(msel,:),'int16');
   for i=1:16;
      field = sprintf('/data/datasetEIT/protocol/Stim.I.%02d(A)',i);
      h5createwrite(field,stimprot(i,:),'int8');
      field = sprintf('/data/datasetEIT/protocol/Meas.V.%02d(V)',i);
      h5createwrite(field,measprot(i,:),'int8');
   end
   h5createwrite('/data/datasetEIT/protocol/Stim.I.freq(Hz)',freq,'double');

   h5createwrite('/data/datasetECG/Meas.V.Abs',ecg_breathhold,'int16');
   h5createwrite('/data/datasetECG/protocol/Stim.I.01(A)',0,'int8');
   h5createwrite('/data/datasetECG/protocol/Stim.I.02(A)',0,'int8');
   h5createwrite('/data/datasetECG/protocol/Stim.I.freq(Hz)',0,'double');
   h5createwrite('/data/datasetECG/protocol/Meas.V.01(V)', 1,'int8');
   h5createwrite('/data/datasetECG/protocol/Meas.V.02(V)',-1,'int8');
   h5createwrite('/data/datasetECG/protocol/Meas.V.freq(Hz)',0,'double');
   h5createwrite('/data/datasetECG/protocol/Meas.Dtime(s)',FR/16,'double');

function h5createwrite(DS, data, datatype);
   fname = 'montreal_data_1995_breathold.h5';
   sz = size(data);
   if strcmp(datatype,'string');
      ver = eidors_obj('interpreter_version')
      if ~ver.isoctave && ver.ver>=10 %only in newer versions
         sz=1;
         data = convertCharsToStrings(data);
      else
         data(data>255) = 'e'; % we know this string, it only has e acute
         data = uint8(data);
         datatype='uint8'; 
         sz=length(data);
      end
   end
   h5create(fname,DS,sz, 'Datatype', datatype);
   h5write(fname,DS, full(data));
