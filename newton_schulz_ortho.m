function [Vperp, P] = newton_schulz_ortho(V, nIters, scale, preserveEnergy)
%NEWTON_SCHULZ_ORTHO  Per-frequency orthogonalisation, Eqs. (17) and (20).
%
%   [Vperp, P] = NEWTON_SCHULZ_ORTHO(V, nIters, scale, preserveEnergy)
%
%   V   : K x F complex matrix, the one-sided spectra of the K modes, i.e. the
%         vectors v(w) of Eq. (18) stacked over frequency.
%   The routine builds one K x K matrix P from the frequency-integrated Gram
%   matrix (Eq. 19) and applies it identically at every frequency
%   (v_perp(w) = P v(w), Eq. 20), which is the Fourier-domain form of the
%   functional update m_k = sum_j C_kj m_j (Eq. 17c).
%
%   preserveEnergy = false reproduces Eqs. (17a)-(17d) verbatim: the modes are
%   divided by (sum_j ||m_j||^2)^(1/2) and the iteration drives G -> I, so the
%   output is orthonormal and no longer sums to f.
%   preserveEnergy = true (default) applies the same orthogonalising map but
%   restores each mode's original L2 norm afterwards, which keeps the projection
%   on the same scale as the Wiener step.  It changes only the scaling, not the
%   directions being orthogonalised.

if nargin < 2 || isempty(nIters),  nIters = 20;   end
if nargin < 3 || isempty(scale),   scale  = 1;    end
if nargin < 4 || isempty(preserveEnergy), preserveEnergy = true; end

K   = size(V, 1);
G   = (V * V') * scale;                       % Eq. (16)
dOld  = sqrt(real(diag(G)));                  % per-mode L2 norms
total = sqrt(real(trace(G)));                 % Eq. (17a)
if total < 1e-15
    Vperp = V;  P = eye(K);  return;
end

P  = eye(K) / total;
Vn = V / total;
for m = 1:nIters
    Gm = (Vn * Vn') * scale;                  % Eq. (17d)
    C  = 1.5*eye(K) - 0.5*Gm;                 % Eq. (17b)
    Vn = C * Vn;                              % Eq. (17c)
    P  = C * P;
end

if preserveEnergy
    dNew = sqrt(real(diag((Vn * Vn') * scale)));
    s = ones(K,1);
    idx = dNew > 1e-15;
    s(idx) = dOld(idx) ./ dNew(idx);
    Vn = bsxfun(@times, s, Vn);
    P  = bsxfun(@times, s, P);
end
Vperp = Vn;
end
