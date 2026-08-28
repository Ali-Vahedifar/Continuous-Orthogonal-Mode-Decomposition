function [modes, omega, info] = vmd_baseline(f, K, alphaVMD, tau, nIter)
%VMD_BASELINE  Classical VMD (Dragomiretskiy & Zosso, 2014).
%
%   Same machinery as COMD with beta = 0 and no Newton-Schulz projection, i.e.
%   the ablation "C-OMD minus the orthogonality constraint".
%   alphaVMD follows the original convention (weight on the bandwidth term);
%   internally alpha_paper = 1/alphaVMD.
if nargin < 2 || isempty(K),        K = 3;          end
if nargin < 3 || isempty(alphaVMD), alphaVMD = 2000; end
if nargin < 4 || isempty(tau),      tau = 0;        end
if nargin < 5 || isempty(nIter),    nIter = 500;    end

o = struct('K', K, 'alpha', 1/alphaVMD, 'beta', 0, 'nsIters', 0, ...
           'tau_lambda', tau, 'tau_gamma', 0, 'nIter', nIter, ...
           'update', 'gauss_seidel');
[modes, omega, info] = comd(f, o);
end
