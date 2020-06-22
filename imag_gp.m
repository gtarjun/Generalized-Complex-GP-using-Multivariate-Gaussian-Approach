function [Y_imag_mu,Y_imag_var,f_imag_mu,f_imag_var,hyp2]=imag_gp(sf_imag,l_imag,sn_imag,X_tr,Y_imag_tr,X_test)

X_real_tr = real(X_tr);
X_imag_tr = imag(X_tr);
X_tr_imag = [X_real_tr X_imag_tr];

meanfunc = {@meanSum, {@meanLinear, @meanConst}}; 
covfunc={'covSum',{@covSEiso, @covConst}};


hyp1.mean = [];  
hyp1.SE= [log(sf_imag);log(l_imag)];
hyp1.const=log(sqrt(1));
hyp1.cov=[hyp1.SE;hyp1.const];
likfunc = @likGauss;   
hyp1.lik = log(sn_imag);
hyp2=hyp1;

hyp2 = minimize(hyp1,@gp,-1000,@infLOO,[],covfunc,likfunc,X_tr_imag,Y_imag_tr );
exp(hyp2.lik)

%nlml2 = gp(hyp2, @infExact, [], covfunc, likfunc, X_tr_2, Y_imag_tr);%calculating negetive log likelihood

[Y_imag_mu,Y_imag_var,f_imag_mu,f_imag_var]=gp(hyp2,@infGaussLik,[],covfunc,likfunc,X_tr_imag,Y_imag_tr,X_test);%calculating the mean and variance


end 