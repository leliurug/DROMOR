% Number of trials
N_trials = 1000;
avg_e1 = 0; avg_e2 = 0; avg_e3 = 0;

for trial = 1:N_trials
    % Random alpha in a reasonable range
    alpha = 0.04 + 0.01*rand();

    % System parameters
    n = 30;
    dt = 0.01;
    m = 5;
    r = 4; r_b = 4;
    T_sim = 100;  % total steps

    % Get system matrices
    [A, B] = heat1d_discrete(alpha, n, dt, m);
    C = [1, zeros(n-1, 1)'];
    D = zeros(1,m);

    % Step 1: DRMOR
    rho = 1;
    epsilon = 1e-8;
    amp = 1;
    Q_bar = amp * eye(m);

    yalmip('clear');
    Q_delta = sdpvar(m, m, 'symmetric');
    EQ = sdpvar(m, m, 'symmetric');
    constraints = [trace(Q_delta + 2*Q_bar - 2*EQ) <= rho^2];
    block_matrix = [sqrtm(Q_bar)*Q_delta*sqrtm(Q_bar) + Q_bar^2, EQ;
                    EQ', eye(m)];
    constraints = [constraints, block_matrix >= 0];
    options = sdpsettings('solver', 'mosek', 'verbose', 0);
    optimize(constraints, -trace(Q_delta), options);
    beta_star = value(trace(Q_delta));
    yalmip('clear');

    Z1_D = sdpvar(r, r, 'symmetric');
    Z_D = blkdiag(Z1_D, zeros(n - r));
    P1_D = sdpvar(n, n, 'symmetric');
    gam = sdpvar(1);
    Psi = A*P1_D*A' - P1_D + B*(Q_bar + beta_star*eye(m))*B';
    con2 = A*(P1_D - Z_D)*A' - (P1_D -Z_D);
    constraints = [Z1_D - epsilon*eye(r)>= 0, P1_D - epsilon*eye(n) >= 0, ...
                   Psi<= -epsilon * eye(n), con2 <= -epsilon * eye(n)];
    constraints = [constraints, P1_D - Z_D >= epsilon*eye(n)];
    constraints = [constraints, trace(C*(P1_D - Z_D)*C') <= gam, gam >= 0];
    optimize(constraints, gam, options);
    P1_D = value(P1_D); Z1_D_opt = value(Z1_D);
    [U, Tz] = schur(Z1_D_opt);
    P2_D = [U; zeros(n-r, r)];
    P3_D = inv(Tz);
    S = inv(P2_D' * inv(P1_D) * P2_D * inv(P3_D));
    hat_A = S * P2_D' * inv(P1_D) * A * P2_D * inv(P3_D);
    hat_B = S * P2_D' * inv(P1_D) * B;
    hat_C = C * P2_D * inv(P3_D);

    % Balanced Truncation
    sys = ss(A, B, C, D, 1);
    sysr = balred(sys, r_b);
    Ar = sysr.A; Br = sysr.B; Cr = sysr.C;

    % Krylov Model Reduction
    [Ak, Bk, Ck, ~] = krylov_mor_discrete(A, B, C, D, r);

    % --- Simulation ---
    y = zeros(1,T_sim); hat_y = zeros(1,T_sim); tilde_y = zeros(1,T_sim); krylov_y = zeros(1,T_sim);
    x = zeros(n,T_sim); hat_x = zeros(r,T_sim); tilde_x = zeros(r,T_sim); krylov_x = zeros(r,T_sim);
    e1 = zeros(1,T_sim); e2 = zeros(1,T_sim); e3 = zeros(1,T_sim);
    mu = zeros(m,1);
    Sigma = eye(m) + diag([1, 0, 0, 0, 0]);

    for k = 1:T_sim-1
        ut = mvnrnd(mu, Sigma)';
        x(:,k+1) = A*x(:,k) + B*ut;
        y(k) = C*x(:,k);
        hat_x(:,k+1) = hat_A * hat_x(:,k) + hat_B * ut;
        hat_y(k) = hat_C * hat_x(:,k);
        tilde_x(:,k+1) = Ar * tilde_x(:,k) + Br * ut;
        tilde_y(k) = Cr * tilde_x(:,k);
        krylov_x(:,k+1) = Ak * krylov_x(:,k) + Bk * ut;
        krylov_y(k) = Ck * krylov_x(:,k);
        e1(k) = norm(y(k) - hat_y(k));
        e2(k) = norm(y(k) - tilde_y(k));
        e3(k) = norm(y(k) - krylov_y(k));
    end

    avg_e1 = avg_e1 + mean(e1);
    avg_e2 = avg_e2 + mean(e2);
    avg_e3 = avg_e3 + mean(e3);
end

avg_e1 = avg_e1 / N_trials;
avg_e2 = avg_e2 / N_trials;
avg_e3 = avg_e3 / N_trials;

fprintf('Average error (Robust Reduction): %.6f\n', avg_e1);
fprintf('Average error (Balanced Truncation): %.6f\n', avg_e2);
fprintf('Average error (Krylov Subspace): %.6f\n', avg_e3);

