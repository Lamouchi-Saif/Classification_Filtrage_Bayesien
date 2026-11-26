function w = filtre_wiener(CC, sigma2)
% FILTRE_WIENER  Coefficients du filtre de Wiener (deconvolution, retard 0).
%
%   w = filtre_wiener(CC, sigma2)
%
%   Minimise J(w) = E[(d(n) - w'x(n))^2] avec x(n) = CC*D(n) + v(n).
%   Comme d(n) est iid de variance 1 (Rd = I) et v(n) blanc :
%       R_x = CC*CC' + sigma2*I           (autocorrelation de x)
%       p   = E[x(n) d(n)] = CC(:,1)      (intercorrelation, = [c(0);0;...;0])
%       w   = R_x^{-1} p
%   La decision est d_hat = sign(w'x) et l'hyperplan de Wiener est {x | w'x = 0}.

    P = size(CC, 1);
    Rx = CC * CC.' + sigma2 * eye(P);
    p  = CC(:, 1);
    w  = Rx \ p;
end
