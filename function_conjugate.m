function h=function_conj(x,v,gamma)
h=v*exp(-conj(x).*x/gamma);
end