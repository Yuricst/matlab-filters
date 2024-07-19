% Example for UKF with two-body problem

clear; close all; clc;
rng(1);

% handle to function that propagates
fun_propagate = @(tspan,x0) propagate_twobody(tspan,x0);

% test function
x0 = [7000; 200; 0; 0; 6.8; 0.8];
t0 = 0.0;
tspan = [0, 3*3600];
[xf,sol] = fun_propagate(tspan,x0);
xs_hat = deval(sol, linspace(tspan(1),tspan(end),100));

sigma_r0 = 10;
sigma_v0 = 0.001;
sigmas0 = [sigma_r0,sigma_r0,sigma_r0,sigma_v0,sigma_v0,sigma_v0]';
P0 = diag(sigmas0.^2);

% % plot
% figure;
% plot3(xs(1,:),xs(2,:),xs(3,:));
% grid on; box on;

% initialize UKF
sigma_dyn = 1e-4;
fun_proc_noise = @(DeltaT) process_noise(DeltaT,sigma_dyn);

x0_hat = x0 + sqrt(diag(P0)) .* randn(6,1);
UKF = UnscentedKalmanFilter(t0,x0_hat,P0,fun_propagate,fun_proc_noise);

sigma_meas = 1.0/3;   % km
meas_interval = 500;
maxiter = 20;
storage.sol_predict = cell(1,maxiter);

storage.P_update = cell(1,maxiter+1);
storage.P_update{1,1} = P0;
storage.sigma_diag_update = zeros(6,maxiter+1);
storage.sigma_diag_update(:,1) = sqrt(diag(P0));
storage.ts_update = [t0];

storage.t_itm = []
storage.sigma_diag_hist = [];

storage.meas = zeros(3,maxiter);

f = waitbar(0,'Please wait...');

% propagate true
tspan = [t0 t0 + maxiter*meas_interval];
[~,sol_true] = fun_propagate(tspan,x0);

for k = 1:maxiter
    % call predict
    [UKF,t_itm,sigma_diag_hist] = predict(UKF,meas_interval);

    % store predict
    storage.sol_predict{1,k} = UKF.sols{1,1};
    storage.t_itm = [storage.t_itm t_itm];
    storage.sigma_diag_hist = [storage.sigma_diag_hist sigma_diag_hist];
    
    % call update
    R = diag([sigma_meas,sigma_meas,sigma_meas].^2);
    fun_meas_model = @(x) pos_meas_model(x);
    y = generate_pos_meas(deval(sol_true,UKF.t), R);

    %UKF = update(UKF,fun_meas_model,y,R);
    C = [eye(3) zeros(3,3)];
    UKF = update_linear(UKF,C,y,R);

    storage.meas(:,k) = y;
    
    % Update waitbar and message
    waitbar(k/maxiter,f,'Recursing UKF');

    % store update
    storage.P_update{1,k+1} = UKF.P;
    storage.ts_update = [storage.ts_update UKF.t];
    storage.sigma_diag_update(:,k+1) = sqrt(diag(UKF.P));
end
close(f);

%% Plot trajectory
dt_plot = 30;
figure('Position',[300,100,700,600])
for k = 1:maxiter
    sol_ = storage.sol_predict{1,k};
    nsteps = max(3,(sol_.x(end) - sol_.x(1))/dt_plot);
    ts = linspace(sol_.x(1), sol_.x(end), nsteps);

    xs_hat = deval(sol_, ts);
    xs_true = deval(sol_true, ts);
    plot3(xs_true(1,:), xs_true(2,:), xs_true(3,:), '-k', 'LineWidth', 2.0);
    hold on;
    plot3(xs_hat(1,:), xs_hat(2,:), xs_hat(3,:), '-b');
end
grid on; box on; axis equal;

%% Plot state history
figure('Position',[100,100,900,600])
tiledlayout(2,3);
axlabels = {'$\delta x$, km', '$\delta y$, km', '$\delta z$, km', ...
            '$\delta v_x$, km/s', '$\delta v_y$, km/s', '$\delta v_z$, km/s'};
for iax = 1:6
    nexttile(iax);
    xlabel('Time, sec', 'Interpreter', 'latex');
    ylabel(axlabels{iax}, 'Interpreter', 'latex');
    set(gca,'fontsize',15);
end

for k = 1:maxiter
    sol_ = storage.sol_predict{1,k};
    nsteps = max(3,(sol_.x(end) - sol_.x(1))/dt_plot);
    ts = linspace(sol_.x(1), sol_.x(end), nsteps);

    xs_hat = deval(sol_, ts);
    xs_true = deval(sol_true, ts);
    error = xs_hat - xs_true;

    for iax = 1:6
        nexttile(iax);
        hold on; box on; grid on;
        plot(ts, error(iax,:),'-b');
    end
end

% covariances
opts={'EdgeColor', 'none',...
  'FaceColor', 'red', 'FaceAlpha', 0.2};
for iax = 1:6
    nexttile(iax)
    sig3 = 3 * storage.sigma_diag_hist(iax,:);
    y1 = -sig3;
    y2 = sig3;
    [y1handle, y2handle, h] = fill_between(storage.t_itm, y1, y2, y1<y2, opts{:});
end

% measurements
xs_true_at_meas = deval(sol_true, storage.ts_update(2:end));
for iax = 1:3
    ys_error = storage.meas(iax,:) - xs_true_at_meas(iax,:);
    error_bars = sigma_meas * ones(size(ys_error));
    nexttile(iax)
    errorbar(storage.ts_update(2:end),ys_error, error_bars, ".", ...
        'Color', 'k')
end

%% functions for UKF
function dx = eom_twobody(t,x,MU)
    dx = [x(4:6); -MU/norm(x(1:3))^3 * x(1:3)];
end

function [xf, sol] = propagate_twobody(tspan, x0)
    MU = 398600.44;
    odeopts = odeset('RelTol',1e-12,'AbsTol',1e-12);
    sol = ode45(@(t,x) eom_twobody(t,x,MU), tspan, x0, odeopts);
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