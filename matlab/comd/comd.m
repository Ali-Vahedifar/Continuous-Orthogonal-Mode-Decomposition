function [modes, omega, info] = comd(f, opts)
%COMD  Continuous-Orthogonal Mode Decomposition.
%
%   [modes, omega, info] = COMD(f, opts)
%
%   Literal implementation of the equations in
%   "Continuous Orthogonal Mode Decomposition: Haptic Signal Prediction in
%    Tactile Internet".  Equation numbers below refer to the paper:
%
%       (11) mode update (Wiener filter)   (12) numerator A
%       (13) centre-frequency update       (14)(15) dual ascent
%       (16) Gram matrix                   (17) Newton-Schulz
%       (20) per-frequency orthogonalisation
%
%   INPUT
%     f     : real vector, the signal to decompose.
%     opts  : struct, all fields optional
%       .K            number of modes                          (default 3)
%       .alpha        reconstruction fidelity weight, Eq. (8)   (default 5e-4)
%                     NOTE alpha_paper = 1/alpha_vmd, so alpha_vmd = 2000 -> 5e-4
%       .beta         orthogonality penalty weight              (default = alpha)
%       .tau_lambda   dual step for lambda,   Eq. (14)          (default 0)
%       .tau_gamma    dual step for Gamma,    Eq. (15)          (default 0)
%       .nIter        maximum ADMM iterations                   (default 500)
%       .tol          relative-change stopping rule             (default 1e-7)
%       .nsIters      Newton-Schulz steps per ADMM iteration    (default 20)
%       .update       'gauss_seidel' | 'jacobi'                 (default gauss_seidel)
%       .relax        damping of the Wiener step (<=0.7 for jacobi)  (default 1)
%       .init         'uniform' | 'random' | 'zero' | 'manual'  (default uniform)
%       .omegaInit    K centre frequencies (cycles/sample) for 'manual'
%       .dc           keep mode 1 at DC                         (default false)
%       .mirror       mirror-extend the signal                  (default true)
%       .normalize    scale f to unit L2 norm internally        (default true)
%       .preserveEnergy  see newton_schulz_ortho                (default true)
%
%   OUTPUT
%     modes : K x N matrix, modes sorted by ascending centre frequency
%     omega : K x 1 centre frequencies in cycles/sample (multiply by fs for Hz)
%     info  : struct with .iterations .reconRelError .gramTime .omegaHistory
%
%   Example
%       fs = 1000; t = (0:999)/fs;
%       f  = cos(2*pi*8*t) + 0.6*cos(2*pi*40*t) + 0.35*cos(2*pi*52*t);
%       [m, w] = comd(f, struct('K',3));
%
%   See also NEWTON_SCHULZ_ORTHO, VMD_BASELINE, DEMO_COMD.

if nargin < 2, opts = struct(); end
d = @(s,v) getfielddef(opts, s, v);
K            = d('K', 3);
alpha        = d('alpha', 5e-4);
beta         = d('beta', alpha);
tau_lambda   = d('tau_lambda', 0);
tau_gamma    = d('tau_gamma', 0);
nIter        = d('nIter', 500);
tol          = d('tol', 1e-7);
nsIters      = d('nsIters', 20);
updateRule   = d('update', 'gauss_seidel');
relax        = d('relax', 1.0);
initMode     = d('init', 'uniform');
dcFlag       = d('dc', false);
mirrorFlag   = d('mirror', true);
normFlag     = d('normalize', true);
preserveE    = d('preserveEnergy', true);

f = double(f(:)).';                       % row vector
N = numel(f);

% ---- internal scaling ---------------------------------------------------
% beta multiplies squared energies while alpha multiplies squared amplitudes,
% so the pair is only meaningful on a fixed signal scale.
fscale = 1.0;
if normFlag
    fscale = norm(f);
    if fscale <= 0, fscale = 1.0; end
end
f = f / fscale;

% ---- mirror extension (as in VMD) --------------------------------------
if mirrorFlag
    h = floor(N/2);
    x = [fliplr(f(1:h)), f, fliplr(f(N-h+1:N))];
else
    x = f;
end
T  = numel(x);
dw = 1/T;                                  % Parseval quadrature weight

freqs = (1:T)/T - 0.5 - 1/T;               % normalised frequency axis
fhat  = fftshift(fft(x));
fhatP = fhat;  fhatP(1:T/2) = 0;           % analytic signal: drop w < 0

posIdx = (T/2+1):T;                        % non-negative frequencies

% ---- centre-frequency initialisation ------------------------------------
switch lower(initMode)
    case 'manual', omega = double(opts.omegaInit(:)).';
    case 'uniform', omega = (0.5/K) * (0:K-1);
    case 'random',  omega = sort(0.5*rand(1,K));
    otherwise,      omega = zeros(1,K);
end
if dcFlag, omega(1) = 0; end

uhat   = zeros(K, T);
lamhat = zeros(1, T);
Gamma  = zeros(K, K);
omegaHistory = zeros(nIter, K);
res = Inf; it = 0;

for it = 1:nIter
    uPrev = uhat;
    G = (uhat(:,posIdx) * uhat(:,posIdx)') * dw;         % Eq. (16)

    uNew = zeros(K, T);
    if strcmpi(updateRule, 'jacobi')                     % fully parallel over k
        tot = sum(uhat, 1);
        for k = 1:K
            corr = zeros(1, T);
            for j = 1:K
                if j == k, continue; end
                corr = corr + (beta*G(k,j) + 0.5*Gamma(k,j)) * uhat(j,:);
            end
            A = fhatP - (tot - uhat(k,:)) + lamhat/(2*alpha) - corr/alpha;   % Eq.(12)
            uNew(k,:) = A ./ (1 + 2*(freqs - omega(k)).^2 / alpha);          % Eq.(11)
        end
    else                                                  % Gauss-Seidel
        for k = 1:K
            others = sum(uNew(1:k-1,:), 1) + sum(uhat(k+1:K,:), 1);
            corr = zeros(1, T);
            for j = 1:K
                if j == k, continue; end
                if j < k, src = uNew(j,:); else, src = uhat(j,:); end
                corr = corr + (beta*G(k,j) + 0.5*Gamma(k,j)) * src;
            end
            A = fhatP - others + lamhat/(2*alpha) - corr/alpha;              % Eq.(12)
            uNew(k,:) = A ./ (1 + 2*(freqs - omega(k)).^2 / alpha);          % Eq.(11)
        end
    end

    if relax ~= 1.0
        uNew = (1-relax)*uhat + relax*uNew;
    end
    uNew(:, 1:T/2) = 0;                                   % stay analytic

    % ---- Eq. (13): centre frequencies -----------------------------------
    for k = 1:K
        p = abs(uNew(k,posIdx)).^2;
        s = sum(p);
        if dcFlag && k == 1
            omega(k) = 0;
        elseif s > 0
            omega(k) = sum(freqs(posIdx) .* p) / s;
        end
    end

    % ---- Eqs. (17)/(20): explicit orthogonal projection ------------------
    if nsIters > 0
        uNew(:,posIdx) = newton_schulz_ortho(uNew(:,posIdx), nsIters, dw, preserveE);
    end

    % ---- Eqs. (14)/(15): dual ascent ------------------------------------
    if tau_lambda ~= 0
        lamhat = lamhat + tau_lambda * (fhatP - sum(uNew,1));
        lamhat(1:T/2) = 0;
    end
    if tau_gamma ~= 0
        Gn = real((uNew(:,posIdx) * uNew(:,posIdx)') * dw);
        Gamma = Gamma + tau_gamma * (Gn - diag(diag(Gn)));
    end

    uhat = uNew;
    omegaHistory(it,:) = omega;
    res = sum(sum(abs(uhat - uPrev).^2)) / (sum(sum(abs(uPrev).^2)) + 1e-30);
    if it > 1 && res < tol, break; end
end

% ---- sort by centre frequency ------------------------------------------
[omega, ord] = sort(omega);
uhat = uhat(ord, :);

% ---- one-sided spectra -> real time-domain modes ------------------------
full = zeros(K, T);
full(:, T/2+1:T)   = uhat(:, T/2+1:T);
full(:, 2:T/2+1)   = conj(uhat(:, T:-1:T/2+1));
full(:, 1)         = conj(full(:, end));
modes = real(ifft(ifftshift(full, 2), [], 2));

if mirrorFlag
    h = floor(N/2);
    modes = modes(:, h+1:h+N);
end

recon = sum(modes, 1);
info = struct();
info.iterations     = it;
info.residual       = res;
info.reconRelError  = norm(recon - f) / (norm(f) + 1e-30);
info.omegaHistory   = omegaHistory(1:it, :);
info.fscale         = fscale;

modes = modes * fscale;
info.gramTime = modes * modes.';
omega = omega(:);
end

% -------------------------------------------------------------------------
function v = getfielddef(s, name, def)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = def;
end
end
