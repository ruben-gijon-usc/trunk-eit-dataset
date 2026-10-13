d = dir;
idx = cell2mat({d(:).isdir});
d = d(idx);

ver

for i = 1:numel(d)
   if strcmp(d(i).name,'..')
      continue
   end
%    fprintf('%s\n', d(i).name);
   cd(d(i).name);
   p = which('eidors_var_id');
   try
      id = eidors_var_id([]);
      if strcmp(id,'id_DA39A3EE5E6B4B0D3255BFEF95601890AFD80709'); %sha1
         fprintf('%s: CORRECT\n', p);
      elseif strcmp(id,'id_99AA06D3014798D86001C324468D497F8C010DD2'); % xxHash of []
         fprintf('%s: CORRECT\n', p);
      else
         fprintf('%s: WRONG\n', p);
      end
   catch err
      msg = textscan(err.message,'%s',1,'delimiter','\r\n');
      fprintf('%s: ERROR: %s\n', p, msg{1}{1});
   end
   if ~strcmp(d(i).name,'.')
      cd ..
   end
end
