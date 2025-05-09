function [A, B] = mass_spring_damper_state_space(M_vals, control_dim)
    % Input: M_vals - vector of masses [M1, M2, ..., Mn] 
    % Input: control_dim_control dimension
    % Output: A, B - state-space matrices

    stiffness_factor = 1.2;   % Increase responsiveness
damping_factor = 0.75;   

    n = length(M_vals);          % Number of masses
    M = diag(M_vals);            % Mass matrix
    M_inv = diag(1 ./ M_vals);   % Inverse mass matrix

    % Construct stiffness matrix K (tridiagonal)
    e = ones(n, 1);
    K = stiffness_factor *  spdiags([-e 2*e -e], -1:1, n, n);
    
    % Construct damping matrix C (3/4 * K)
    C = damping_factor * K;

    % State-space A matrix: [0 I; -M^{-1}K -M^{-1}C]
    A = [zeros(n), eye(n);
         -M_inv * K, -M_inv * C];

    % Input matrix B: [0; M^{-1}]
    B = [zeros(2*n- control_dim, control_dim);
         M_inv(end-control_dim+1: end, end-control_dim+1: end)];
end