function [modes, omega, info] = none_baseline(f, K)
%NONE_BASELINE  No-decomposition baseline.
%
%   [modes, omega, info] = NONE_BASELINE(f, K)
%
%   Tests whether decomposing into modes (COMD/VMD_BASELINE/SVMD) helps at
%   all, versus feeding the network the raw signal directly. There is
%   nothing to decompose into multiple channels here by construction, so K
%   must be 1.
%
%   Same (modes, omega, info) return shape as COMD/VMD_BASELINE/SVMD so it is
%   a drop-in replacement wherever those are used.

if nargin < 2 || isempty(K), K = 1; end
if K ~= 1
    error('none_baseline:K', 'the no-decomposition baseline only makes sense with K=1');
end

f = double(f(:)).';
modes = f;
omega = 0;
info = struct('iterations', 0, 'reconRelError', 0);
end
