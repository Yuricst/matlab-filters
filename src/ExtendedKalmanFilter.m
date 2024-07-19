%EXTENDEDKALMANFILTER Summary of this class goes here
%   The EKF is a nonlinear estimator.
%
classdef ExtendedKalmanFilter

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
        function obj = ExtendedKalmanFilter(t0,x0,P0,...
                fun_propagate, fun_process_noise)
            %ExtendedKalmanFilter Construct an instance of this class
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
            %predict  Time prediction of EKF

            % integrate mean and STM
            tspan = [obj.t, obj.t + DeltaT];
            [xf,sol] = obj.fun_propagate(tspan, obj.xhat);

            % store predicted state estimate
            obj.xhat = xf(1:6);

            % store predicted covariance
            Phi = reshape(xf(7:end), [6,6]);
            obj.P = Phi * obj.P * transpose(Phi) + obj.fun_process_noise(DeltaT);
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

    end
end
