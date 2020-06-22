close all;clear all;clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%addpath('L:\Gaussian Process Formulation\gpml-matlab-v4.2-2018-06-11\startup.m')
%Data Generation
v_r=0.1; 
gamma_r=0.6
N=100;
L=6;
x1=linspace(-L,L,N);
x2=1j*linspace(-L,L,N);
X=repmat(x1,N,1)+repmat(x2,N,1).';
 
 
h1=function1(X,v_r,gamma_r);
v_j=0.05; 
gamma_j=1.5;

h2=function1(X,v_j,gamma_j);
 
Sr=randn(N,N);
Sj=randn(N,N);
 
U=conv2(Sr,h1+1j*h2)+conv2(Sj,h1+1j*h2);  %Original function 
Un=U+0.0562*randn(size(U))+1j*0.0562*randn(size(U)); %Original function w/ noise


z1=linspace(-2*L,2*L,2*N-1);
delta = z1(2)-z1(1);
T=round(5/delta); % We crop around the origin from -5 to +5  
%Z1: Center at position N
%U: center at position (N,N)
z2=z1(N-T:N+T);
U2n=Un(N-T:N+T,N-T:N+T);  %cropped function around origin with noise 
U2=U(N-T:N+T,N-T:N+T);    %cropped function 


%training samples
n=size(U2n);
sx=round(0.5+rand(1,500)*n(1)); %Indexes of the samples in x
sy=round(0.5+rand(1,500)*n(1)); %Indexes of the samples in y
%Training samples are taken at z2(sx) and z2(sy)
X_tr=z2(sx)+1j*z2(sy);X_tr=X_tr.';
index=sub2ind(size(U2),sx,sy);
Y_tr=U2n(index).'; % Values of the function at sampled positions
Y_real_tr = real(Y_tr);
Y_imag_tr = imag(Y_tr);


%Xtest
x1=linspace(-2*L,2*L,2*N-1);
x2=1j*linspace(-2*L,2*L,2*N-1);
X_test_1=repmat(x1,length(x1),1)+repmat(x2,length(x1),1).';

X_test=X_test_1(:); 
X_test_2 = [real(X_test) imag(X_test)];
X_test_2_i = [imag(X_test) real(X_test)];



meanfunc = {@meanSum, {@meanLinear, @meanConst}}; 
covfunc={'covSum',{@covSEiso, @covConst}};


hyp.mean = [];  
hyp.SE= [log(sweep_noise);log(1)];
hyp.const=log(sqrt(1));
hyp.cov=[hyp.SE;hyp.const];
likfunc = @likGauss;
   
hyp.lik = log(sn);
hyp2 = minimize(hyp, @gp, -100, @infExact, [], covfunc, likfunc, X_tr_2_i,Y_imag_tr );
exp(hyp2.lik)

nlml2 = gp(hyp2, @infExact, [], covfunc, likfunc, X_tr_2_i, Y_imag_tr);%calculating negetive log likelihood

[Y_mu,Y_imag_var, f_imag_mu, f_imag_var] = gp(hyp2, @infExact, [], covfunc, likfunc, X_tr_2_i, Y_imag_tr, X_test_2);%calculating the mean and variance
