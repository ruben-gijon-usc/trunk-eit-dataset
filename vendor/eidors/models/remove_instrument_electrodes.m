function fmdl = remove_instrument_electrodes(fmdl)
% pre-delete all 'instrument' nodes
% system matrix can't handle them
% NOTE: instrument nodes must be at end
%
% Used by system_mat_instrument

% (C) 2023 Andy Adler. Licence GPL v2 or v3

   if ~isfield(fmdl,'electrode')
      return % nothing to do
   end
   rmelec = [];
   for i=num_elecs(fmdl):-1:1
      try
         enodes = fmdl.electrode(i).nodes;
         if ischar(enodes) && strcmp(enodes,'instrument');
            rmelec(end+1) = i;
         end
      end
   end
   fmdl.electrode(rmelec)= [];

   if ~isfield(fmdl,'stimulation')
      return % nothing to do
   end

   for i=1:length(fmdl.stimulation)
      fmdl.stimulation(i).stim_pattern(rmelec) = []; 
      fmdl.stimulation(i).meas_pattern(:,rmelec) = []; 
      try % if it exists
      fmdl.stimulation(i).volt_pattern(rmelec) = []; 
      end
   end


