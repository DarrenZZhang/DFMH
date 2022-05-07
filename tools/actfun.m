function [H]=actfun(W,X,act_fun)
   switch act_fun 
          case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                H = sigm(W*X);
          case 'tanh'
                H = tanh(W*X);
          case 'max'
                H = max(0,W*X);
          case 'shrink'
                H1=W*X;
                theta=0.1;
                H =(((-theta<H1)+(H1<theta))<1.5).* H1;
          case 'softplus'
                H = log(1+exp(W*X));
          case 'linear'
                H = W*X;
   end

end
