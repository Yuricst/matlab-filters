%UNSCENTEDKALMANFILTER Summary of this class goes here
%   The UKF is a nonlinear estimator using the unscented transform.
%   This is the emplementation of Algorithm 8.17 (page 156) in [1].
%
%   Ref:
%       [1] https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf
%
classdef UnscentedKalmanFilter

    properties
        fun_propagate (1,1)
        fun_process_noise
        t (1,1) double
        xhat (:,1) double
        P (:,:) double
        nx (1,1) double
        res_x (:,:) double
        sols (:,:)
    end
    
    methods
        function obj = UnscentedKalmanFilter(t0,x0,P0,...
                fun_propagate, fun_process_noise)
            %UNSCENTEDKALMANFILTER Construct an instance of this class
            %   Detailed explanation goes here
            
            % checks that inputs are indeed functions
            if ~isa(fun_propagate,'function_handle')
                error(message('MATLAB:integral:funArgNotHandle'));
            end
            if ~isa(fun_process_noise,'function_handle')
                error(message('MATLAB:integral:funArgNotHandle'));
            end

            obj.fun_propagate = fun_propagate;
            obj.fun_process_noise = fun_process_noise;

            obj.t = t0;
            obj.xhat = x0;
            obj.P = P0;

            obj.nx = length(x0);
            
            % initialize
            obj.res_x = zeros(obj.nx,2*obj.nx+1);
            obj.sols = cell(1,obj.nx);
        end
        
        function [obj,t_itm,sigma_diag_hist] = predict(obj,DeltaT,N_intermediate)
            %predict  Time prediction of UKF
            %
            % Inputs:
            %   DeltaT : duration of time prediction
            %
            
            arguments
                obj
                DeltaT (1,1) double
                N_intermediate (1,1) double = 20;
            end

            % 1. sample sigma points, eqn (8.82)
            [X,Wm,Wc] = getSigmaPoints(obj.xhat, obj.P);

            % 2. propagate through nonlinear dynamics, eqn (8.83)
            tspan = [obj.t, obj.t + DeltaT];
            x_pred = zeros(obj.nx,size(X,2));
            for i = 1:size(X,2)
                [x_pred(:,i),obj.sols{1,i}] = obj.fun_propagate(tspan, X(:,i));
            end

            % 3. predict mean, eqn (8.84)
            obj.xhat = x_pred * Wm';

            % 3(pre). compute history of diagonal sigma's for plotting
            if N_intermediate > 1
                sigma_diag_hist = zeros(6,N_intermediate);
                Dt_itm = linspace(0,DeltaT,N_intermediate);
                t_itm = obj.t + Dt_itm;
                for k = 1:N_intermediate
                    P_ = obj.P + obj.fun_process_noise(Dt_itm(k));
                    for i = 1:size(X,2)
                        res_x_ = x_pred(:,i) - obj.xhat;
                        P_ = P_ + Wc(i) * (res_x_ * res_x_');
                    end
                    sigma_diag_hist(:,k) = sqrt(diag(P_));
                end
            else
                t_itm = [];
                sigma_diag_hist = [];
            end

            % 3. predict covariance, eqn (8.84)
            obj.P = zeros(obj.nx,obj.nx) + obj.fun_process_noise(DeltaT);
            for i = 1:size(X,2)
                obj.res_x(:,i) = x_pred(:,i) - obj.xhat;
                obj.P = obj.P + Wc(i) * (obj.res_x(:,i) * obj.res_x(:,i)');
            end
            
            % 4. update time
            obj.t = obj.t + DeltaT;
        end

        function obj = update_linear(obj,C,y,R)
            %update  Linear measurement update
            %If the measurement is linear, s.t. h(x) = Cx,
            %then linear measurement update can be used.
            %
            % Inputs:
            %   C : m-by-n matrix for linear measurement model
            %   y : m-by-1 measurement
            %   R : m-by-m measurement covariance
            %
            CT = transpose(C);
            K = obj.P * CT / (C*obj.P*CT + R);
            obj.xhat = obj.xhat + K * (y - C * obj.xhat);
            I_KC = eye(obj.nx) - K*C;
            obj.P = I_KC * obj.P * transpose(I_KC) + (K*R*transpose(K));
        end

        function obj = update(obj,fun_meas_model,y,R)
            %update  Measurement update with unscented transform
            %The measurement model function is expected to have signature
            %   h = fun_meas_model(xhat)
            %where h is the predicted measurement from state estimate xhat.
            %
            % Inputs:
            %   fun_meas_model : measurement model function
            %   y              : m-by-1 measurement
            %   R              : m-by-m measurement covariance
            %

            [nm,~] = size(R);

            if ~isa(fun_meas_model,'function_handle')
                error(message('MATLAB:integral:funArgNotHandle'));
            end
            
            % 1. sample sigma points, eqn (8.85)
            [X,Wm,Wc] = getSigmaPoints(obj.xhat, obj.P);

            % 2. map through measurmeent model, eqn (8.86)
            y_pred = zeros(nm,size(X,2));
            for i = 1:size(X,2)
                y_pred(:,i) = fun_meas_model(X(:,i));
            end

            % 3. predict measurement mean, eqn (8.87)
            y_mean = y_pred * Wm';

            % 3. predict measurement covariance, eqn (8.87)
            Py = R;
            Pxy = zeros(obj.nx,nm);
            for i = 1:size(X,2)
                res_y = y_pred(:,i) - y_mean;
                Py = Py + Wc(i) * (res_y * res_y');
                Pxy = Pxy + Wc(i) * (obj.res_x(:,i) * res_y');
            end

            % 4. perform Kalman update, eqn (8.88)
            K = Pxy / Py;
            obj.xhat = obj.xhat + K * (y - y_mean);
            obj.P = obj.P - K * Py * K';

        end
    end
end

