% Example for UKF with two-body problem

clear; close all; clc;
rng(1);

MU = 398600.44;
% handle to function that propagates
func_jac = get_func_jac(MU);
fun_propagate_stm = @(tspan,x0) propagate_twobody_stm(tspan,x0,MU,func_jac);

x0 = [7000; 200; 0; 0; 6.8; 0.8];
t0 = 0.0;
tspan = [0, 3*3600];

sigma_r0 = 10;
sigma_v0 = 0.001;
sigmas0 = [sigma_r0,sigma_r0,sigma_r0,sigma_v0,sigma_v0,sigma_v0]';
P0 = diag(sigmas0.^2);

% initialize EKF
sigma_dyn = 1e-4;
fun_proc_noise = @(DeltaT) process_noise(DeltaT,sigma_dyn);

x0_hat = x0 + sqrt(diag(P0)) .* randn(6,1);
EKF = ExtendedKalmanFilter(t0,x0_hat,P0,fun_propagate_stm,fun_proc_noise);
EKF1 = predict(EKF, 300)


%% functions for EKF
function dx = eom_twobody(t,x,MU)
    dx = [x(4:6); -MU/norm(x(1:3))^3 * x(1:3)];
end

function dxstm = eom_twobody_stm(t,x,MU,func_jac)
    dxstm = zeros(42,1);
    dxstm(1:6,1) = eom_twobody(t,x,MU);

    % STM derivative
    Phi = reshape(x(7:end,1), [6,6]);
    A = func_jac(t,x);
    Phidot = A * Phi;
    dxstm(7:end,1) = reshape(Phidot, [36,1]);
end

function [func_A_alias,symA] = get_func_jac(MU)
    % symbolic variables
    x = sym('x',[6,1]);

    % system dynamics
    r = sqrt(x(1)^2 + x(2)^2 + x(3)^2);
    dxdt = [...
        x(4);
        x(5);
        x(6);
        -MU/r^3 * x(1);
        -MU/r^3 * x(2);
        -MU/r^3 * x(3)];

    % compute Jacobian
    symA = jacobian(dxdt, x);
    
    % make it a function
    func_A = matlabFunction(symA);
    func_A_alias = @(t,x) func_A(x(1),x(2),x(3));
end

function [xf, sol] = propagate_twobody_stm(tspan,x0,MU,func_jac)
    odeopts = odeset('RelTol',1e-12,'AbsTol',1e-12);
    x0stm = [x0; reshape(eye(6),[36,1])];
    sol = ode45(@(t,x) eom_twobody_stm(t,x,MU,func_jac), tspan, x0stm, odeopts);
    xf = deval(sol, tspan(end));
end

function Q = process_noise(tf,sigma_dyn)
    Q_rr = 1/3*eye(3) * tf^3;
    Q_rv = 1/2*eye(3) * tf^2;
    Q_vv = eye(3) * tf;
    Q = sigma_dyn^2 * [Q_rr Q_rv; Q_rv Q_vv];
end

function h = pos_meas_model(x)
    h = x(1:3,1);
end

function y = generate_pos_meas(x,R)
    y = x(1:3,1) + sqrt(diag(R)) .* randn(3,1);
end