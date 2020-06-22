
close all;clear all;clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('L:\Gaussian Process Formulation\gpml-matlab-v4.2-2018-06-11\'))

%Data attributes
v_r=0.1; 
gamma_r=0.6;
v_j=0.05; 
gamma_j=1.5;
N=100;
L=6;

%GP Solution 
sf_imag=1;
sf_real=1;
sn_real=0.1;
sn_imag=0.1;
l_imag=1;
l_real=1;

iterations=500;

sn_imag_opt=zeros(iterations,1);
sn_real_opt=zeros(iterations,1);
sf_imag_opt=zeros(iterations,1);
l_imag_opt=zeros(iterations,1);
sf_real_opt=zeros(iterations,1);
l_real_opt=zeros(iterations,1);
squared_error=zeros(iterations,1);
counter=0;
MSE=zeros(length(sn_real),1);
% for k=1:length(sn_real)
%    
    counter=counter+1
for i=1:iterations
    
    
[X_tr,Y_imag_tr,X_test,Y_real_tr,Yn,index,T,z2,Y2,sy,sx,Y_tr]=data_generator(v_r,gamma_r,v_j,gamma_j,N,L);%Generating data for MonteCarlo

evalc('[Y_imag_mu,Y_imag_var,f_imag_mu,f_imag_var,hyp2]=imag_gp(sf_imag,l_imag,sn_imag,X_tr,Y_imag_tr,X_test)');%GP solution for imgainary part

Y_imag_2 = Y_imag_mu(index);%estimated imaginary corresponding to Y training.

evalc('[Y_real_mu,Y_real_var, f_real_mu, f_real_var,hyp4] = real_gp(sf_real,l_real,sn_real,X_tr,Y_imag_mu,Y_real_tr,Y_imag_2,X_test)');%GP solution for real part

Y_real_estimated = reshape(Y_real_mu,length(Yn),length(Yn));
Y_imag_estimated = reshape(Y_imag_mu,length(Yn),length(Yn));

Y_estimated = Y_real_estimated + 1i*Y_imag_estimated;

sn_imag_opt(i)=exp(hyp2.lik);
sn_real_opt(i)=exp(hyp4.lik);
sf_imag_opt(i)=exp(hyp2.SE(1));
l_imag_opt(i)=exp(hyp2.SE(2));
sf_real_opt(i)=exp(hyp4.SE(1));
l_real_opt(i)=exp(hyp4.SE(2));

squared_error(i)=mean(abs(Yn(:)-Y_estimated(:)).^2,1); 
sn_imag_opt(i)
sn_real_opt(i)
sf_imag_opt(i);
l_imag_opt(i);
sf_real_opt(i);
l_real_opt(i);

look=squared_error(i)

end 

MSE = mean(squared_error,1)
% end 

Y_estimated_c = Y_estimated(N-T:N+T,N-T:N+T);

% %representation
% figure(1)
% mesh(z2,z2,real(Y2))
% hold on
% plot3(z2(sy),z2(sx),real(Y_tr),'.')
% hold off
% 
% figure(2)
% mesh(z2,z2,imag(Y2))
% hold on
% plot3(z2(sy),z2(sx),imag(Y_tr),'.')
% hold off
% 
% figure(3)
% plot(z2,real(Y2(:,0.5*(size(Y2,1)-1))),'r');
% hold on
% plot(z2,real(Y_estimated_c(:,0.5*(size(Y_estimated_c,1)-1))),'k');
% 
% figure(4)
% mesh(z2,z2,real(Y_estimated_c))
% 
% figure(5)
% mesh(z2,z2,imag(Y_estimated_c))



 
 



