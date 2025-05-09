function [A, B] = random_schur_stable_system(n, m)
    % Generates a Schur stable A (n x n) and random B (n x m)

    % Generate stable eigenvalues inside unit circle
    eigenvalues = rand(n, 1) * 0.95;  % all < 1 in magnitude
    D = diag(eigenvalues);

    % Random orthogonal matrix via QR
    [Q, ~] = qr(randn(n));

    % Construct A = Q*D*Q' -> Schur stable and real
    A = Q * D * Q';

    % Generate random B
    B = randn(n, m);
end
