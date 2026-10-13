function out = subtract_rank(in, ranklim)
% SUBTRACK_RANK: subtract a ranked level from data
%
%  imgout = subtract_rank(imgin, 0.3)
%    subtract the 30% percentile from imgr.elem_data
%  imgout = subtract_rank(imgin, [0.1,0.3])
%    subtract an average of 10-30% percentile from imgr.elem_data
%
% if input is a 'image' object, then operate on elem_data. Otherwise operate on value

% (C) 2024 Andy Adler. Licence GPL v2 or v3
% $Id: subtract_rank.m 7124 2024-12-29 15:18:32Z aadler $ 

if ischar(in) && strcmp(in,'UNIT_TEST'); do_unit_test; return; end

if isstruct(in) && strcmp(in.type, 'image') 
   use_elem_data = true;
   out = in;
   in = in.elem_data;
elseif isnumeric(in)
   use_elem_data = false;
else
   error('Unknown type for subtract_rank');
end

[~,idx] = sort(sum(in,1));
ll = length(idx);
lim= round(ranklim([1,end])*ll);
idx_= idx(lim(1):lim(2));

sub= in - mean(in(:,idx_),2);
if use_elem_data
  out.elem_data = sub;
else
  out  = sub;
end

function do_unit_test
  ed = [1:10] + [0;3];
  x = subtract_rank(ed,.3);
  expect = [0;0] + [-2:7];
  unit_test_cmp('subtract_rank1',x,expect);
  x = subtract_rank(ed,[.1,.3]);
  expect = [0;0] + [-1:8];
  unit_test_cmp('subtract_rank2',x,expect);

  img = eidors_obj('image','','elem_data',ed);
  iout = subtract_rank(img,[.1,.3]);
  unit_test_cmp('subtract_rank3',iout.elem_data,expect);
