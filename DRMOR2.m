
r = 3;
r_b =r;
amp = 1;

% Discrete-time system matrices
% A = [0.8, 0.2, 0, 0; 
%      0, 0.7, 0.4, 0;
%      0, 0, 0.6, 0.4;
%      0,  0,  0,  0.8];
% B =  [1  0 ; 1  0 ; 0  1 ; 0  1];


alpha = 0.001;   % thermal diffusivity
    n = 100;         % number of spatial interior points
    dt = 0.001;     % time step
    m = 5;          % number of input points
    T = 100.0;        % total simulation time

    % Get system matrices
    [A, B] = heat1d_discrete(alpha, n, dt, m);

C = [1, zeros(n-1, 1)'];
D = zeros(1,m);

n = size(A,1);
m = size(B,2); % Input Order

rho = 1; % Uncertainty bound

epsilon = 10e-8;
yalmip('clear')
%% Step 1: Solve for β*
tic
Q_bar =   amp * eye(m); % Example nominal covariance matrix
% Define variables QΔ and EQ as symmetric matrices
Q_delta = sdpvar(m, m, 'symmetric');
EQ = sdpvar(m, m, 'symmetric');
% Define constraints
constraints = [trace(Q_delta + 2*Q_bar - 2*EQ) <= rho^2];
block_matrix = [sqrtm(Q_bar)*Q_delta*sqrtm(Q_bar) + Q_bar^2, EQ;
                EQ', eye(m)];
constraints = [constraints, block_matrix >= 0];
% Solve optimization
options = sdpsettings('solver', 'mosek', 'verbose', 1, 'mosek.MSK_DPAR_INTPNT_CO_TOL_PFEAS', 1e-2);
optimize(constraints, -trace(Q_delta), options); % Maximize tr(QΔ)
beta_star = value(trace(Q_delta));
Q_delta_opt = value(Q_delta);
yalmip('clear');
%% Step 2: Solve for γ, P_D, Z_D
Z1_D = sdpvar(r, r, 'symmetric');
Z_D = blkdiag(Z1_D, zeros(n - r)); % Enforce Z_D structure
P1_D = sdpvar(n, n, 'symmetric');
gam = sdpvar(1);
% Define Ψ matrix (replace with system-specific Lyapunov equation)
Psi = [A*P1_D*A' - P1_D + B*(Q_bar + beta_star*eye(m))*B'];
con2 = A*(P1_D - Z_D)*A' - (P1_D -Z_D);
% Define constraint
constraints = [Z1_D - epsilon*eye(r)>= 0, P1_D - epsilon*eye(n) >= 0, Psi<= -epsilon * eye(n), con2 <= -epsilon * eye(n)];
constraints = [constraints, P1_D - Z_D >= epsilon*eye(n)];
constraints = [constraints, trace(C*(P1_D - Z_D)*C') <= gam, gam >= 0];
% Solve optimization
options = sdpsettings('solver', 'mosek', 'verbose', 0, ...
    'debug', 1, ...
    'mosek.MSK_DPAR_INTPNT_TOL_REL_GAP', 1e-1, ...
    'mosek.MSK_DPAR_INTPNT_TOL_PFEAS', 1e-1, ...
    'mosek.MSK_DPAR_INTPNT_TOL_DFEAS', 1e-1, ...
    'mosek.MSK_DPAR_INTPNT_CO_TOL_REL_GAP', 1e-1, ...
    'mosek.MSK_IPAR_INTPNT_MAX_ITERATIONS', 2);
optimize(constraints, gam, options);
P1_D = value(P1_D);
Z_D = value(Z_D);
Z1_D_opt = value(Z1_D);
Psi_opt =value(Psi);
gamma_star = value(gam);

%% Step 3: Schur decomposition of Z1_D
[U, T] = schur(Z1_D_opt);
 Z1_D = Z1_D_opt; % Update Z1_D
%% Step 4: Construct P2_D and P3_D
P2_D = [U; zeros(n-r, r)];
P3_D = T^(-1);
%% Step 5: write the reduced order system dynamics
S = (P2_D' * P1_D^(-1) * P2_D * P3_D^(-1))^(-1);
hat_A = S * P2_D'*inv(P1_D)*A * P2_D*inv(P3_D); 
hat_B = S * P2_D'*inv(P1_D)*B;
hat_C = C * P2_D * inv(P3_D);
yalmip('clear')
toc
%% Comparsion Balanced Truncation
tic
%% Comparsion Balanced Truncation

% Compute Gramians
X = dlyap(A, B*Q_bar*B') + 10e-5 * eye(n);    % Controllability Gramian
Y = dlyap(A', C'*C) + 10e-5 * eye(n);     % Observability Gramian

% Cholesky factorization
R = chol(X, 'lower');     % X = R*R'
L = chol(Y, 'lower');     % Y = L*L'

% Compute SVD of L'*R
[U, S, V] = svd(L' * R);
HSV = diag(S);            % Hankel Singular Values


% Partition singular values/vectors
U1 = U(:, 1:r_b);
S1 = S(1:r_b, 1:r_b);
V1 = V(:, 1:r_b);

% Balancing transformation
T = R * V1 * diag(1./sqrt(diag(S1)));
Tinv = diag(1./sqrt(diag(S1))) * U1' * L';

% Transform to balanced coordinates
A_bal = Tinv * A * T;
B_bal = Tinv * B;
C_bal = C * T;

% Truncate to keep dominant states
tilde_A = A_bal(1:r_b, 1:r_b);
tilde_B = B_bal(1:r_b, :);
tilde_C = C_bal(:, 1:r_b);

toc
% Or use reduce directly (more modern):

tic
[tilde2_A, tilde2_B, tilde2_C, tilde2_D] = krylov_mor_discrete(A, B, C, D, r);
toc
%% system run
T = 10e1;
y = zeros(1,T);
e1 = zeros(1,T);
e2 = zeros(1,T);
e3 = zeros(1,T);
hat_y = zeros(1,T);
x = zeros(n,T);
u = zeros(m,T);
hat_x = zeros(r,T);
mu = zeros(m,1);  % 均值向量
Sigma =  amp * eye(m) + diag([1, zeros(1,m-1)]);  % 协方差矩阵
tilde_y = zeros(1,T);
tilde_x = zeros(r,T);

tilde2_y = zeros(1,T);
tilde2_x = zeros(r,T);


for k = 1: T-1
%u(:,k) = mvnrnd(mu, Sigma)';
ut = mvnrnd(mu, Sigma)';
    x(:,k+1) = A*x(:,k) + B * ut;
    y(k) = C*x(:,k);
    hat_x(:,k+1) = hat_A*hat_x(:,k) + hat_B * ut;
    hat_y(k) = hat_C * hat_x(:,k);
    tilde_x(:,k+1) = tilde_A * tilde_x(:,k) + tilde_B* ut;
    tilde_y(k) = tilde_C * tilde_x(:,k);
    tilde2_x(:,k+1) = tilde2_A * tilde2_x(:,k) + tilde2_B* ut;
    tilde2_y(k) = tilde2_C * tilde2_x(:,k);
    e1(k) = norm(y(k)-hat_y(k));
    e2(k) = norm(y(k)- tilde_y(k));
    e3(k) = norm(y(k)- tilde2_y(k));
end

figure
plot(y)
hold on
plot(hat_y)
plot(tilde_y)
plot(tilde2_y)
figure
plot(e1)
hold on
plot(e2)
plot(e3)