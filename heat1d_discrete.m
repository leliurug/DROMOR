function [A, B] = heat1d_discrete(alpha, n, dt, m)
    % Parameters
    dx = 1 / (n + 1);            % domain [0,1], n interior points
    lambda = alpha * dt / dx^2; % CFL number

    % Construct tridiagonal A
    e = ones(n, 1);
    A = spdiags([lambda*e, (1 - 2*lambda)*e, lambda*e], -1:1, n, n);

    % Input matrix B: heat inputs at m positions
    % Here we inject input at m equally spaced positions
% Construct B = [zeros(n - m, m); eye(m)]
    input_indices = round(linspace(1, n, m));
    B = zeros(n, m);
    for j = 1:m
        B(input_indices(j), j) = 1;
    end
end
