clear;clear memory;
addpath('./tools')
dataname = 'mvCifar10';
nbits_set = [16];
%[8,16,32,48,64,96,128];
%result number 10
%% Load dataset
load('mvCifar10.mat')
for it = 1:3
    Dis = EuDist2(X{it},Anchor{it},0);
    sigma = mean(mean(Dis)).^0.5;
    feavec = exp(-Dis/(2*sigma*sigma));
    X{it} = bsxfun(@minus, feavec', mean(feavec',2));
end

view_num = size(X,2);
data_our.gnd = gnd+1;
gnd = gnd+1;
% Separate Train and Test Index
for n_iters = 1:1
tt_idx = [];
for ind = 1:10
    list = find(ind==gnd);
    tt_idx = [tt_idx; randsample(list , 100)];
end
list = 1:numel(gnd);
list(tt_idx) = [];
tr_idx = list; 
ttgnd = gnd(tt_idx);
trgnd = gnd(tr_idx);
% clear gnd;

for ii=1:length(nbits_set)
    nbits = nbits_set(ii);
    data_our.indexTrain= tr_idx;
    data_our.indexTest= tt_idx;
    ttfea = cell(1,view_num);
    for view = 1:view_num
        data_our.X{view} = normEqualVariance(X{view}')';
        ttfea{view} = data_our.X{view}(:,tt_idx);
    end

        pars.beta       = 10; % parameters\lambda.5
        pars.gamma    = 0.01; % parameters\gamma1000
        pars.lambda = 0.1;% parameters\eta.1
        pars.Iter_num = 4;
        pars.nbits    = nbits;
        pars.r = 3;
        
        [B_trn,U1,U2,U3, W, U_W, R, alpha, trtime] = DFMH_fun(data_our,pars);
        
        % for testing
        H = zeros(nbits,length(ttgnd));
        for ind = 1:size(ttfea,2)
            H = H+alpha(ind)*U3{ind}'*U2{ind}'*U1{ind}'*ttfea{ind};
        end
        B_tst = H'*U_W >0;
        
        WtrueTestTraining = bsxfun(@eq, ttgnd, trgnd');

        %% Evaluation
        B1 = compactbit(B_trn);
        B2 = compactbit(B_tst);
        DHamm = hammingDist(B2, B1);
        [~, orderH] = sort(DHamm, 2);
        MAP = calcMAP(orderH, WtrueTestTraining);
        fprintf('iter = %d, Bits: %d, MAP: %.4f...   \n', n_iters, nbits, MAP);
  
end

end
