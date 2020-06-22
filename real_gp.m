function [Y_real_mu,Y_real_var, f_real_mu, f_real_var,hyp4] = real_gp(sf_real,l_real,sn_real,X_tr,Y_imag_mu,Y_real_tr,Y_imag_2,X_test)

X_real_tr = real(X_tr);
X_imag_tr = imag(X_tr);
X_tr_real = [X_real_tr X_imag_tr Y_imag_2]; % training matrix for estimation of the real part
X_test_real = [X_test Y_imag_mu]; % test matrix for estimation of the real part

meanfunc = {@meanSum, {@meanLinear, @meanConst}}; 
covfunc={'covSum',{@covSEiso, @covConst}};


hyp3.mean = [];  
hyp3.SE= [log(sf_real);log(l_real)];
hyp3.const=log(sqrt(1));
hyp3.cov=[hyp3.SE;hyp3.const];
likfunc = @likGauss;   
hyp3.lik = log(sn_real);

hyp4 = minimize(hyp3, @gp,-1000,@infLOO,[],covfunc,likfunc,X_tr_real,Y_real_tr );
exp(hyp4.lik)

%nlml2 = gp(hyp4,@infExact,[],covfunc,likfunc,X_tr_real,Y_real_tr);%calculating negetive log likelihood

[Y_real_mu,Y_real_var, f_real_mu, f_real_var] = gp(hyp4, @infGaussLik, [], covfunc, likfunc, X_tr_real, Y_real_tr, X_test_real);%calculating the mean and variance

end