function [X_tr,Y_imag_tr,X_test,Y_real_tr,Yn,index,T,z2,Y2,sy,sx,Y_tr]=data_generator(v_r,gamma_r,v_j,gamma_j,N,L)

x1=linspace(-L,L,N);
x2=1j*linspace(-L,L,N);
X=repmat(x1,N,1)+repmat(x2,N,1).';
  
h1=function_conjugate(X,v_r,gamma_r);
h2=function_conjugate(X,v_j,gamma_j);
 
Sr=randn(N,N);%real noise
Sj=randn(N,N);%imaginary noise
 
Y=conv2(Sr,h1+1j*h2)+conv2(Sj,h1+1j*h2);  %Original function 
Yn=Y+0.1*randn(size(Y))+1j*0.01*randn(size(Y)); %Original function w/ noise


z1=linspace(-2*L,2*L,2*N-1);
delta = z1(2)-z1(1);
T=round(5/delta); % We crop around the origin from -5 to +5  
%Z1: Center at position N
%U: center at position (N,N)
z2=z1(N-T:N+T);
Y2n=Yn(N-T:N+T,N-T:N+T);  %cropped function around origin with noise 
Y2=Y(N-T:N+T,N-T:N+T);    %cropped function

Yn_real = real(Yn(:));
Yn_imag = imag(Yn(:));

l = length(Yn_real);

%training samples

n=size(Y2n);

sx=round(0.5+rand(1,500)*n(1)); %Indexes of the samples in x
sy=round(0.5+rand(1,500)*n(1)); %Indexes of the samples in y
%Training samples are taken at z2(sx) and z2(sy)
X_tr=z2(sx)+1j*z2(sy);X_tr=X_tr.';
index=sub2ind(size(Y2),sx,sy);
Y_tr=Y2n(index).'; % Values of the function at sampled positions
Y_real_tr = real(Y_tr);
Y_imag_tr = imag(Y_tr);


%Xtest
x1=linspace(-2*L,2*L,2*N-1);
x2=1j*linspace(-2*L,2*L,2*N-1);
X_test_1=repmat(x1,length(x1),1)+repmat(x2,length(x1),1).';
X_test_2=X_test_1(:); 
X_test=[real(X_test_2) imag(X_test_2)];

end
