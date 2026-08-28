function [modes, omega, info] = svmd(f, K, alphaVMD, tau, nIter, tol)
%SVMD  Successive Variational Mode Decomposition.
%
%   [modes, omega, info] = SVMD(f, K, alphaVMD, tau, nIter, tol)
%
%   Reference: Nazari & Sakhaei, "Successive Variational Mode Decomposition",
%   Signal Processing 174:107610, 2020.
%
%   HONESTY NOTE (read before trusting this as a literal transcription): the
%   published SVMD extracts modes one at a time, and each step's objective
%   adds a *second* penalty term beyond the usual VMD bandwidth term -- a
%   constraint on the leftover residual that discourages it from also being
%   narrowband around the just-claimed center frequency, so each step's
%   Wiener filter differs from a plain single-mode VMD solve. That
%   residual-shaping term is NOT reproduced here verbatim: reconstructing its
%   exact closed form from memory, without the paper's equations in front of
%   it, risks silently encoding a wrong formula and calling it faithful --
%   worse than being explicit about the gap.
%
%   What IS unambiguous, and is what this function implements, is the
%   "successive" structure itself: unlike VMD/C-OMD, which solve for all K
%   modes jointly, SVMD solves K independent single-mode problems in
%   sequence, each on the residual left over from the previous step. That is
%   implemented exactly, by reusing this repo's own exact VMD_BASELINE solver
%   with K=1 at each step -- itself a valid single-mode VMD/C-OMD solve,
%   since a 1-mode system has no cross-mode terms to get wrong.
%
%   If you have the paper's exact per-step equations, paste them and this
%   becomes a literal transcription the same way COMD is for the C-OMD paper.
%
%   INPUT   same convention as VMD_BASELINE, applied once per extracted mode.
%   OUTPUT  modes : K x N, ordered by ascending centre frequency
%           omega : K x 1 centre frequencies (cycles/sample)
%           info  : struct with .iterations (K x 1) and .reconRelError
%
%   See also VMD_BASELINE, COMD.

if nargin < 2 || isempty(K),        K = 3;          end
if nargin < 3 || isempty(alphaVMD), alphaVMD = 2000; end
if nargin < 4 || isempty(tau),      tau = 0;        end
if nargin < 5 || isempty(nIter),    nIter = 500;    end
if nargin < 6 || isempty(tol),      tol = 1e-7;     end

f = double(f(:)).';
residual = f;
N = numel(f);
modes = zeros(K, N);
omega = zeros(K, 1);
iters = zeros(K, 1);

for k = 1:K
    [m, w, stepInfo] = vmd_baseline(residual, 1, alphaVMD, tau, nIter);
    modes(k, :) = m;
    omega(k)    = w;
    iters(k)    = stepInfo.iterations;
    residual    = residual - m;
end

[omega, ord] = sort(omega);
modes = modes(ord, :);
iters = iters(ord);

info = struct();
info.iterations    = iters;
info.residualEnergy = sum(residual.^2);
info.reconRelError  = norm(f - sum(modes, 1)) / (norm(f) + 1e-30);
end
