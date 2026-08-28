function demo_comd()
%DEMO_COMD  Verification demo: C-OMD vs VMD on a signal with overlapping bands.
%
%   Reproduces the property the paper is built on -- VMD is orthogonal only as a
%   side effect of disjoint spectral support, so as soon as two modes overlap it
%   leaves residual correlation, while C-OMD's explicit projection removes it at
%   no cost in reconstruction error.
%
%   Run:  >> demo_comd
%
%   Every number printed below is computed here and now; nothing is hard-coded.

thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'comd'), fullfile(thisDir, 'vmd'), ...
        fullfile(thisDir, 'svmd'), fullfile(thisDir, 'none'));

fs = 1000;  N = 1000;  t = (0:N-1)/fs;
f  = cos(2*pi*8*t) + 0.6*cos(2*pi*40*t) + 0.35*cos(2*pi*52*t);
omegaInit = [8 40 52]/fs;         % identical initialisation for both methods

fprintf('\nSignal: 8 Hz + 0.6*40 Hz + 0.35*52 Hz, fs = %g Hz, K = 3\n', fs);
fprintf('%-14s %6s %14s %16s   %s\n', 'method','iters','recon rel err', ...
        'max |corr_ij|', 'centre freqs (Hz)');
fprintf('%s\n', repmat('-', 1, 78));

o = struct('K',3,'init','manual','omegaInit',omegaInit);

ov = o; ov.beta = 0; ov.nsIters = 0; ov.alpha = 1/2000;
[mV, wV, iV] = comd(f, ov);
report('VMD', mV, wV, iV, fs);

for ns = [1 5 20]
    oc = o; oc.nsIters = ns;
    [mC, wC, iC] = comd(f, oc);
    report(sprintf('C-OMD ns=%d', ns), mC, wC, iC, fs);
end

% ---- reconstruction check ----------------------------------------------
oc = o; oc.nsIters = 20;
[mC, ~, ~] = comd(f, oc);
fprintf('\nmax |f - sum_k m_k|  : %.3e   (signal amplitude %.3f)\n', ...
        max(abs(f - sum(mC,1))), max(abs(f)));

if usejava('jvm') || exist('OCTAVE_VERSION', 'builtin')
    try
        figure('Name','C-OMD demo');
        subplot(4,1,1); plot(t, f); ylabel('f(t)'); title('signal and C-OMD modes');
        for k = 1:3
            subplot(4,1,k+1); plot(t, mC(k,:));
            ylabel(sprintf('m_%d', k));
        end
        xlabel('time (s)');
    catch
        fprintf('(plotting skipped: no display)\n');
    end
end
end

function report(name, m, w, info, fs)
G = m*m.';  d = sqrt(diag(G));  C = G ./ (d*d.');
off = abs(C - eye(numel(d)));  off(logical(eye(numel(d)))) = 0;
fprintf('%-14s %6d %14.3e %16.3e   %s\n', name, info.iterations, ...
        info.reconRelError, max(off(:)), mat2str(round(w'*fs*100)/100));
end
