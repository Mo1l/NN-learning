clear all  
%% Layer 0 - initializing data %%
Xt = dlmread("C:\Users\nikol\OneDrive\8. semester\Machine Learning\Asignment 6\sincTrain25.dt");
Xv = dlmread("C:\Users\nikol\OneDrive\8. semester\Machine Learning\Asignment 6\sincvalidate10.dt");

X = [ones(size(Xt,1),1), Xt(:,1)]';         % Data with the addition of a constant
Y = Xt(:,2)';

Xval = [ones(size(Xv,1),1),Xv(:,1)]';
Yval = Xv(:,2)';
d = 20;         % Depth of the neural network

rng(42)                                  % Seed
a = randn(d,2)*0.1%ones(d,2);            % Hidden layer parameters
W = randn(d+1,1)*0.1%ones(d+1,1);        % output layer parameters

%% Part 1
% Analytically computed Error derivatives
n = size(X,2)
for i =1:n
    Wn(:,i) = NNEstimation(X(:,i),Y(:,i), d, a, W)
end
dEdW = sum(Wn,2)

%Wn = NNEstimation(Xtest,Ytest, d, a, W)

%Numerically computed derivatives

En = @(Y,Yhat) norm(Y - Yhat)^2

% This is EW - Benchmark
for i=1:n
    [Z,S,Ypred] = fwprop(X(:,i),d,a,W)
    Ev(:,i) = En(Y(:,i),Ypred)
end
E = 0.5*sum(Ev)

% This is f(X+eps*e)
h =sqrt(eps)*0.5
Wimp = [a(:,1);a(:,2);W]
R = size(Wimp,1)

% attempt at making a loop over every variable to get numerical derivatives
for i=1:R 
% Decomposing vector to loop over
Wv = Wimp(i,1) + h;
Wh = Wimp;
Wh(i,1) = Wv;

for j =1:size(X(:,1),1)
    if j == 1
        ao(:,j) = Wh(1:d);
    else 
        ao(:,j) = Wh(d*(j-1) + 1:d*2);      % Hidden layer vector 
    end
end

Wo = Wh(d*size(X(:,1),1) + 1:end);          % W output vector

% Calculating f(w + h*e_i)
    for j=1:n
        [Z,S,Ypred] = fwprop(X(:,j),d,ao,Wo);
        Evx(i,j) = En(Y(:,j),Ypred);
    end

Ex(i,1) = 0.5*sum(Evx(i,:));
end

out_part1_dEdWnum=(Ex - E)/h
dEdW
out_part1_change = dEdW - out_part1_dEdWnum

%% Part 2
nn = 600
% Implementing gradient descent
% Initializition
rho = 0.0018;
for i =1:n
    Wn(:,i) = NNEstimation(X(:,i),Y(:,i), d, a, W);
end
Gradient(:,1) = sum(Wn,2)                       % Initial gradient
Weight(:,1) = constructweight(a,W)              % initial weights
%Weight(:,2) = constructweight(a,W).*2            % Will be overridden but is neccessary to get the loop running
i=2;
K = size(X(:,1),1)
stop = 1
for i=2:nn
    %if i ==2 | stop > h  
        % Training on Training set
        Weight(:,i) = Weight(:,i-1) - rho.*Gradient(:,i-1);  % Weight adjustment
        [aest, West] = deconstructweight(K,d,Weight(:,i));       % Weight deconstr
        for j =1:n
            [Wn(:,j),Ypred(:,j)] = NNEstimation(X(:,j),Y(:,j), d, aest, West);
            Ev(:,j) = En(Y(:,j),Ypred(:,j));
        end
        Gradient(:,i) = sum(Wn,2);                      % Initial gradient
        Etra(:,i) = sum(Ev)/size(X,2);
end
normgradienttra =  sqrt(sum(Gradient.^2,1));

%% Validation 
for i=1:size(Weight,2)
        [aval, Wval] = deconstructweight(K,d,Weight(:,i));       % Weight deconstr
        for j =1:10%size(Xval,2)
            [Wnval(:,j),Ypredval(:,j)] = NNEstimation(Xval(:,j),Yval(:,j), d, aval, Wval);
            Evalx(:,j) = En(Yval(:,j),Ypredval(:,j));
        end
        Eval(:,i) = sum(Evalx)/size(Xval,2);
        Gradientval(:,i) = sum(Wnval,2);                      % Initial gradient

end
normgradientval =  sqrt(sum(Gradientval.^2,1));

%% Plots
width=800;
height=300;

figure (1)
set(gcf,'position',[200,200,width,height]);
title('Training and validation norm','FontSize',10);
plot(1:nn,normgradienttra,1:nn,normgradientval);
legend('Training','Validation');
ylabel('Gradient norm'); % y-axis label
xlabel('iterations'); % y-axis label
grid on;
 legend('location','northeast');
%print -dpng figuretestnorm, 'C:\Users\nikol\OneDrive\8. semester\Machine Learning\Asignment 6';

 
figure (2)
set(gcf,'position',[200,200,width,height]);
title('Training and Validation error','FontSize',10);
plot(1:nn,Etra,1:nn,Eval);
legend('Training','Validation');
set(gca, 'YScale', 'log')
ylabel('Mean-squared error'); % y-axis label
xlabel('iterations'); % y-axis label
grid on;
 legend('location','northeast');

%print -dpng figure015TESTerror, 'C:\Users\nikol\OneDrive\8. semester\Machine Learning\Asignment 6';

figure (3)
set(gcf,'position',[200,200,width,height]);
title('Training and Validation error','FontSize',10);
plot(1:nn,Etra,1:nn,Eval);
legend('Training','Validation');
ylabel('Mean-squared error'); % y-axis label
xlabel('iterations'); % y-axis label
grid on;
 legend('location','northeast');

% Fitting the plot
Wopt = Weight(:,end)
T(1,1)=-15;
i=2
while T(1,i-1)<15
    T(1,i) = T(1,i-1)+0.05;
    i=i+1
end
dat = [ones(1,size(T,2));T]
i=2

while T(1,i-1)<15
    [aplot,Wplot] =deconstructweight(2,d,Wopt)
    [Zcccc, Scccc, Yplotfit(1,i)] = fwprop(dat(:,i-1), d, aplot, Wplot)
    i=i+1
end

sinc = @(x) sin(x)./x;
yplotreal = sinc(T)

figure (3)
set(gcf,'position',[200,200,width,height]);
title('fitted model','FontSize',10);
plot(T,Yplotfit,T,yplotreal);
legend('Training','real');
ylabel('f(x)'); % y-axis label
xlabel('x'); % y-axis label
grid on;
 legend('location','northeast');
%print -dpng figure015TEStfit, 'C:\Users\nikol\OneDrive\8. semester\Machine Learning\Asignment 6';










%% Function Library
function Wimp =constructweight(a,W)
Wimp = [a(:,1);a(:,2);W]
end

% Inputs dimension of data, number of h neurons, Weights(vectorform)
% outputs (a) matrix and W vector 
function [ao,Wo] =deconstructweight(n,d,Wh)
for j =1:n
    if j == 1
        ao(:,j) = Wh(1:d);
    else
        ao(:,j) = Wh(d*(j-1) + 1:d*2);      % Hidden layer vector
    end
end
Wo = Wh(d*n + 1:end);          % W output vector
end


% Does forward and backpropagation and computes the derivatives 
function [Wn, Ypred] = NNEstimation(Xtest,Ytest, d, a, W)
    [Z, S, Ypred] =fwprop(Xtest,d,a,W);
    delta = backprop(Ypred, Ytest, S, W, d);
    [w1,w2] = derivatives(delta, Z, Xtest, d);
    Wn = [w1(:,1); w1(:,2); w2];
end

% Computes the derivatives of the error function for each parameter (weight)
function [outh1, outo] =derivatives(delta,Z,Xtest,d)
% for output layer weights
outo = delta(end,1).*Z;

% for hidden layer weights
n = size(Xtest,1)
    for i=1:n
        outh1(:,i) = delta(1:d).*Xtest(i,1);
    end
end


% Computes the Backpropagation algorithm
% Input: output node value, true value, weights for output node, no. nodes
function deltaout = backprop(Ypred, Y, S, W, d, n)       
% Backpropagation %%

% derivative of activation function
hprime = @(S) 1./((1 + abs(S))^2);

% Computing delta:
% output layer
deltaend = Ypred - Y;

% Computing the deltas from layer hidden layer deltas: delta_i = h'(a_i)
for j=1:d
    delta(j,1)= hprime(S(j,1))*W(j+1,1)*deltaend;
end

deltaout = [delta; deltaend];


% Now we want to calculate all the derivatives There should be 
%K*d + (d+1)=31 for K=2 and d = 10
%output layer
end


% Computes forward propagation
function [Zout, Sout, Yout] = fwprop(X, d, a, W)       % Data and dimension
% layer 1 %%

% Constructing S  input for activation function% 
S = @(x,a) a*x;
% P = S(x,a)
% activation function %
% ones(size(X,1),1)
h = @(S) S./(ones(size(S,1),1) + abs(S));

% Output of layer 1 %
% H = h(P) %h(S(x,a))          % function input x and parameters a
 
%Output layer %%
out = @(w,in) w'*in;    % function input from layer 
%out(W,H)

%Forward propagation %%
    Sout = S(X,a);              % Layer 1 - S
    Zout = [1; h(S(X,a))];      % layer 1 - Activation
    Yout = out(W,Zout);             % output layer
end

    
    