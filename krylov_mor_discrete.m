function [Ar, Br, Cr, Dr] = krylov_mor_discrete(A, B, C, D, r, tol)
% Krylov MOR for discrete-time systems with specified reduced order.
% Inputs:
%   A, B, C, D: System matrices (discrete-time)
%   r:          Desired reduced order
%   tol:        (Optional) Tolerance for rank estimation, default=1e-12
% Outputs:
%   Ar, Br, Cr, Dr: Reduced-order model matrices
%   V:              Projection matrix (truncated to r columns if needed)

if nargin < 6
    tol = 1e-12; % Default tolerance
end

n = size(A, 1);
m = size(B, 2);
V = []; % Initialize orthonormal basis

% Initial block: Orthonormalize B
[Q, R] = qr(B, 0);
s = diag(R);
current_rank = sum(abs(s) > tol * abs(s(1)));
V = Q(:, 1:current_rank);

if isempty(V)
    error('B has no significant column space. MOR failed.');
end

% Continue building subspace until desired order is reached
while size(V, 2) < r
    % Current block (last set of vectors added)
    current_block = V(:, end - current_rank + 1 : end);
    
    % Apply A to current block
    W = A * current_block;
    
    % Modified Gram-Schmidt against all previous vectors
    for k = 1:size(V, 2)
        v = V(:, k);
        h = v' * W;
        W = W - v * h;
    end
    
    % Orthonormalize the new block
    [Q, R] = qr(W, 0);
    s = diag(R);
    new_rank = sum(abs(s) > tol * abs(s(1)));
    Q = Q(:, 1:new_rank);
    
    if new_rank == 0
        break; % No more vectors to add
    end
    V = [V, Q]; % Append new orthonormal vectors
    current_rank = new_rank;
end

% Truncate to exact desired order if necessary
if size(V, 2) > r
    V = V(:, 1:r);
end

% Warn if desired order not achieved
if size(V, 2) < r
    warning('Desired order %d not achieved. Using %d instead.', r, size(V, 2));
end

% Compute reduced-order model
Ar = V' * A * V;
Br = V' * B;
Cr = C * V;
Dr = D;
end
