%getSigmaPoints Compute sigma points given the estimate and covariance matrix
% This function implements the sigma points according to Särkkä, S. (2007).
%
% Ref:
%   Särkkä, S. (2007). On unscented Kalman filtering for state estimation of 
%   continuous-time nonlinear systems. IEEE Transactions on Automatic Control,
%   52(9), 1631–1641. https://doi.org/10.1109/TAC.2007.904453
%
% For further description of the unscented transform, see pg.153 in :
%   https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf
% Equation numbers are also from this pdf.
%
% Syntax:   [X,Wm,Wc] = getSigmaPoints(x,P)
%           [X,Wm,Wc] = getSigmaPoints(_,'alpha',5e-3)
% 
% Inputs:
%   x     : N-by-1 matrix of estimate of the state
%   P     : N-by-N matrix of covariance of the state
%   alpha : (optional) parameter dictating spread of sigma points around the
%           mean (usually choose small value, default: 1e-3)
%   beta  : (optional) parameter for higher order moments (default: 2.0)
%   kappa : (optional) tuning parameter (default: 0.0)
%
% Outputs:
%   X  : N-by-(2N+1) matrix of sigma points
%   Wm : 1-by-(2N+1) matrix of weights for propagated mean
%   Wc : 1-by-(2N+1) matrix of weights for propagated covariance
%
function [X,Wm,Wc] = getSigmaPoints(x,P,options)
    
    arguments
        x (:,1) double
        P (:,:) double
        % optional arguments
        options.alpha (1,1) double = 1e-3
        options.beta  (1,1) double = 2.0
        options.kappa (1,1) double = 0.0
    end
    
    % number of states
    n = length(x);
    assert(size(P,1) == n, "First dimension of P is not n!");
    assert(size(P,2) == n, "First dimension of P is not n!");
    
    % compute sigma points
    X  = zeros(n,2*n+1);
    Wm = zeros(1,2*n+1);
    Wc = zeros(1,2*n+1);
    
    % compute scaling factor, eqn (8.74)
    lambda = options.alpha^2 * (n + options.kappa) - n;
    
    % compute sigma points, eqn (8.77)
    X(:,1) = x;
    for i = 1:n
        cholP = chol(P); %,'lower');
        X(:,i+1)   = x + sqrt(n + lambda) * cholP(:,i);
        X(:,i+n+1) = x - sqrt(n + lambda) * cholP(:,i);
    end
    
    % compute weights for mean and covariance, eqn (8.75)
    Wm(1) = lambda / (n + lambda);
    Wc(1) = lambda / (n + lambda + 1 - options.alpha^2 + options.beta);
    for i = 1:2*n
        Wm(i+1) = 1 / (2*(n + lambda));
        Wc(i+1) = 1 / (2*(n + lambda));
    end
end
