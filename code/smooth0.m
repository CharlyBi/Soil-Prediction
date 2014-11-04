function [ y ] = smooth0( h, x, dly )
% smooth0 0 delay filtering 
%   pad the input matrix with constant value then filtering.
%   filtering is done over rows.
%   Author: Charly B, 2014
%   
% License:
%   This software is distributed under the GNU General Public License 
%   (version 2 or later); please refer to the file LICENSE.txt, included 
%   with the software, for details. 

a0 = mean(x(1:dly,:));         %average paded value 
a1 = mean(x(end-dly+1:end,:)); %average paded value 
y = [repmat(a0,dly,1); x; repmat(a1,dly,1)];
y = filter(h,1,y);
y = y(2*dly+1:end,:);



end

