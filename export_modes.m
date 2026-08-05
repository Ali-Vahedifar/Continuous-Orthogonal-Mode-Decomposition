function outFile = export_modes(inFile, outFile, opts)
%EXPORT_MODES  Decompose every channel of a trace and save the modes for Python.
%
%   outFile = EXPORT_MODES(inFile, outFile, opts)
%
%   inFile  : .mat or .csv holding a T x C matrix of channels (one column per
%             channel, e.g. human/robot x {F,V,P} x {x,y,z}).
%   outFile : .mat written with variables
%               modes   K x T x C   decomposed modes
%               omega   K x C       centre frequencies (cycles/sample)
%               chNames 1 x C       channel names if the input had a header
%   opts    : passed straight to COMD, plus
%               .method  'comd' (default) | 'vmd'
%               .mode    'online' (default, causal) | 'offline'
%               .buffer  history length for the online mode (default 256)
%               .stride  hop between online decompositions (default 1)
%
%   'online' decomposes the last `buffer` samples at each step and keeps only
%   the newest ones, which is causal and matches the latency the paper charges
%   to inference.  'offline' decomposes the whole trace at once: much faster,
%   but the modes at time t then depend on samples after t, so use it for
%   exploration only.

if nargin < 3, opts = struct(); end
method = getdef(opts, 'method', 'comd');
mode   = getdef(opts, 'mode',   'online');
buffer = getdef(opts, 'buffer', 256);
stride = getdef(opts, 'stride', 1);
K      = getdef(opts, 'K', 3);
opts.K = K;

[~,~,ext] = fileparts(inFile);
chNames = {};
if strcmpi(ext, '.mat')
    S = load(inFile);
    fn = fieldnames(S);
    X = S.(fn{1});
else
    X = dlmread(inFile, ',', 1, 0);
end
[T, C] = size(X);
fprintf('loaded %s : %d samples x %d channels\n', inFile, T, C);

modes = zeros(K, T, C);
omega = zeros(K, C);
for c = 1:C
    x = X(:,c).';
    if strcmpi(mode, 'offline')
        [m, w] = runOne(x, method, opts);
    else
        m = zeros(K, T);
        [m0, w] = runOne(x(1:buffer), method, opts);
        m(:, 1:buffer) = m0;
        tt = buffer;
        while tt < T
            e = min(tt + stride, T);
            [mm, w] = runOne(x(e-buffer+1:e), method, opts);
            m(:, tt+1:e) = mm(:, end-(e-tt)+1:end);
            tt = e;
        end
    end
    modes(:,:,c) = m;
    omega(:,c)   = w;
    fprintf('  channel %d/%d done\n', c, C);
end
save('-v7', outFile, 'modes', 'omega', 'chNames');
fprintf('wrote %s\n', outFile);
end

function [m, w] = runOne(x, method, opts)
if strcmpi(method, 'vmd')
    [m, w] = vmd_baseline(x, opts.K);
else
    [m, w] = comd(x, opts);
end
end

function v = getdef(s, n, d)
if isstruct(s) && isfield(s, n) && ~isempty(s.(n)), v = s.(n); else, v = d; end
end
