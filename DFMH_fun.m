function [B_train,U1,U2,U3, W, U_W, R, alpha, trtime]=DFMH_fun(data, pars)
%set the parameters
gamma     = pars.gamma; 
beta     = pars.beta; 
lambda   = pars.lambda;
Iter_num = pars.Iter_num;
nbits    = pars.nbits;
trIdx = data.indexTrain;
r = pars.r;

view_num = size(data.X,2);
XXT = cell(1,view_num);
dim = zeros(1,view_num);
our_data = cell(1,view_num);
for ind = 1:view_num
    our_data{ind} = data.X{ind}(:,trIdx);
    [dim(ind), n] = size(our_data{ind});
    XXT{ind} = our_data{ind}*our_data{ind}';%X*X'
end

% label matrix Y = c x N   (c is amount of classes, N is amount of instances)
if isvector(data.gnd)
    L_tr = data.gnd(trIdx);
    Y = sparse(1:length(L_tr), double(L_tr), 1); 
    Y = full(Y');
else
    L_tr = data.gnd(trIdx,:);
    Y = double(L_tr');
end

%initialize
d=size(our_data{1},1);

L=300;%[nbits,d]
L2=L;

U1=cell(1,view_num);U2=cell(1,view_num);
for iter = 1:size(our_data,2)
    U1{iter}=randn(d,L);
    U2{iter}=randn(L,L2);
end



U3=cell(1,view_num);

B = randn(nbits, n)>0; B = B*2-1;
W = randn(nbits, size(Y,1));  %G=Y'*B;
H = W*Y;

H1=cell(1,view_num);
H2=cell(1,view_num);
for iter = 1:size(our_data,2)
    H1{iter}=U1{iter}\our_data{iter};
    H2{iter}=U2{iter}\(U1{iter}\our_data{iter});
end

alpha = ones(view_num,1)/view_num;

tic;
%training
for iter = 1:Iter_num
    fprintf('The %d-th iteration...\n',iter);
    alpha_r = alpha.^r;
    [~,dim_y]=size(Y*Y');
        
    %----U-step----
    for ind = 1:size(our_data,2)
        U1{ind}=our_data{ind}*pinv(H1{ind});
        U2{ind}=(pinv(U1{ind})*our_data{ind}) * pinv(H2{ind});
        U3{ind}=(pinv(U1{ind}*U2{ind}))*our_data{ind} * pinv(H);
    end
    
    %----Hi-step----
    for ind = 1:size(our_data,2)
        UTX=U1{ind}'*our_data{ind};
        UTUH=U1{ind}'*U1{ind}*H1{ind};
        M1=(abs(UTX)+UTX)./2;
        M2=(abs(UTUH)-UTUH)./2;
        M3=(abs(UTX)-UTX)./2;
        M4=(abs(UTUH)+UTUH)./2;
        temp=sqrt((M1+M2)./max((M3+M4),1e-10));
        H1{ind}=H1{ind}.*max(temp,1e-10);
        
        UTUTX=U2{ind}'*U1{ind}'*our_data{ind};
        UTUTUUH=(U1{ind}*U2{ind})'*(U1{ind}*U2{ind})*H2{ind};
        N1=(abs(UTUTX)+UTUTX)./2;
        N2=(abs(UTUTUUH)-UTUTUUH)./2;
        N3=(abs(UTUTX)+UTUTX)./2;
        N4=(abs(UTUTUUH)+UTUTUUH)./2;
        temp=sqrt((N1+N2)./max((N3+N4),1e-10));
        H2{ind}=H2{ind}.*max(temp,1e-10);
    end
    
    %----W-step----
    W=beta*B*Y'/(beta*Y*Y'+gamma*eye(dim_y));
    
    %----R-step----
    [Pr,~,Qr] = svd(B*H','econ');
    R=Pr*Qr';
    
    %----H-step----
    Zv=cell(1,view_num);
    for ind =1:size(our_data,2)
        Zv{ind}=U1{ind}*U2{ind}*U3{ind};
    end
    
    [dimh11,dimh12]=size(Zv{1}'*Zv{1});
    temph1=zeros(dimh11,dimh12);
    [dimh21,dimh22]=size(Zv{1}'*our_data{1});
    temph2=zeros(dimh21,dimh22);
    
    for ind =1:size(our_data,2)
        temph1=temph1+alpha_r(ind)*Zv{ind}'*Zv{ind};
        temph2=temph2+alpha_r(ind)*Zv{ind}'*our_data{ind};
    end
    H=(temph1+lambda*(R'*R))\(temph2+lambda*R'*B);
    
    %----B-step----
    A=2*lambda*R*H+2*beta*W*Y;
    B=sign(A);
    
    %----alpha-step----
    h = zeros(view_num,1);
    for view = 1:view_num
        h(view) =norm(our_data{view}-U1{view}*U2{view}*U3{view}*H,'fro')^2;
    end
    Temp=bsxfun(@power,h,1/(1-r));
    alpha=bsxfun(@rdivide,Temp,sum(Temp));
end
trtime=toc;
B_train=B'>0;

%out-of-Sample
H0=zeros(nbits,n);
for ind=1:view_num
    S=alpha_r(ind)*U3{ind}'*U2{ind}'*U1{ind}'*our_data{ind};
    H0=H0+S;
end
NT=(H0*H0' + 1*eye(size(H0,1))) \ H0;
U_W=NT*B_train;
