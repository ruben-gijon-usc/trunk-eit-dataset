function s_out = mergestructs(s_in, varargin)
% MERGESTRUCTS: merge structs where fields in later structs override earlier
%
% struct_out = mergestructs(base, struct1, struct2, ...)
%
% Example:
%   s1 = struct('a',1,'b',2);
%   s2 = struct('c',5,'b',3);
%   mergestructs(s1,s2) => .a=1, .b=3, .c=3
% 
% Note this only processes the top level of a struct tree

% $Id: mergestructs.m 7002 2024-11-24 13:11:35Z aadler $
% (C) 2024 A Adler: License GPL v2 or v3

  if ischar(s_in) && strcmp(s_in,'UNIT_TEST'); do_unit_test; return; end

  s_out = s_in;
 
  for i = 1:length(varargin);
    ssi = varargin{i};
    if ~isstruct(ssi); error('Mergestructs: only struct args'); end
    for fn = fieldnames(ssi)'
      s_out.( fn{1} ) = ssi.( fn{1} ); 
    end
  end

function do_unit_test
  s1 = struct('a',1,'b',2);
  s2 = struct('c',5,'b',3);
  s3 = struct('d',6,'b',4);

  so = mergestructs(s1);
  unit_test_cmp('1 arg',so,s1);

  so = mergestructs(s1,s2);
  so2= struct('a',1,'b',3,'c',5);
  unit_test_cmp('2 arg',so,so2);

  so = mergestructs(s1,s2,s3);
  so3= struct('a',1,'b',4,'c',5,'d',6);
  unit_test_cmp('3 arg',so,so3);
