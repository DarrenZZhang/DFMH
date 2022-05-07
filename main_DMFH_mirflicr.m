clear;clear memory;

addpath('./tools')
% dataname = 'mirflickr';
nbits_set = [32];%[8,16,32,48,64,96,128];
dataname = {'mirflickr'};%[iaprtc12,pascal07,espgame,Corel5k];
for i=1:size(dataname,2)  
load([dataname{i} '.mat']);
fprintf([ dataname{i} ' dataset loaded...\n']);
view_num = size(X,2);
n_anchor = 1000;
Anchor = cell(1,view_num);
n_Sam = size(X{1},1);


for n_iters = 1:1
    
	for it = 1:view_num
	    X{it} = double(X{it});
	    anchor = X{it}(randsample(n_Sam, n_anchor),:);
	    Dis = EuDist2(X{it},anchor,0);
	    sigma = mean(mean(Dis)).^0.5;
	    feavec = exp(-Dis/(2*sigma*sigma));
	    XX{it} = bsxfun(@minus, feavec', mean(feavec',2));
	end

	% Separate Train and Test Index
	tt_num = 1000;
	data_our.gnd = gnd;
	tt_idx = randsample(n_Sam, tt_num);
	list = 1:n_Sam;
	list(tt_idx) = [];
	tr_idx = list; 
	ttgnd = gnd(tt_idx,:);
	trgnd = gnd(tr_idx,:);
	% clear gnd;

	for ii=1:length(nbits_set)
		nbits = nbits_set(ii);
		data_our.indexTrain= tr_idx;
		data_our.indexTest= tt_idx;
		ttfea = cell(1,view_num);
		for view = 1:view_num
		data_our.X{view} = normEqualVariance(XX{view}')';
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
		H = zeros(nbits,tt_num);
		for ind = 1:size(ttfea,2)
		    H = H+alpha(ind)*U3{ind}'*U2{ind}'*U1{ind}'*ttfea{ind};
		end
		B_tst = H'*U_W >0;

		cateTrainTest = zeros(size(trgnd,1), size(ttgnd,1),'uint8');
		for i_con = 1:size(trgnd,2)
		    test = find(ttgnd(:,i_con));
		    train = find(trgnd(:,i_con));
		    cateTrainTest(train, test) = 1;
		end
		WtrueTestTraining = logical(cateTrainTest');

		%% Evaluation
		B1 = compactbit(B_trn);
		B2 = compactbit(B_tst);
	       %% evaluation
		disp('Evaluation of DFMH...');
		DHamm = hammingDist(B2, B1);
		[~, orderH] = sort(DHamm, 2);
		MAP = calcMAP(orderH, WtrueTestTraining);
		fprintf('iter = %d, Bits: %d, MAP: %.4f...   \n', n_iters, nbits, MAP);
	end
end
