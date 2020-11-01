function W = svm_kernel_dd( a, fracrej, para, Kernel_type)
%ELM_KERNEL_DD Extreme Learning Machine Kernel Data Description
%
%   W = ELM_KERNEL_DD(A, FRACREJ, PARA, KERNEL_TYPE)
%
% Optimizes a Extreme learning machine kernel data description for the
% dataset A. KERNEL_TYPE represents the type of adopted kernel.
% KERNEL_TPYE could be: 'RBF_kernel' for RBF kernel; 'Random_kernel' 
% for Random feature mapping using sigmoid activation function; 
% 'Lin_kernel' for Linear kernel; 'Poly_kernel' for Polynomial kernel.
% PARA contains all the hyperparameters. 
% PARA(1) is always the regularization coefficient C.
% When KERNEL_TYPE = 'RBF_kernel', PARA(2) is SIGMA;
% When KERNEL_TYPE = 'Random_kernel', PARA(2) is the number of hidden
% neurons L.
% FRACREJ gives the fraction of the target set which will be rejected.
%
% An example for RBF_kernel:
%     w = elm_kernel_dd(a, 0.1, [power(10,8), 1.41], 'RBF_kernel')   
% An exmple for Poly_kernel:
%     w = elm_kernel_dd(a, 0.1, [power(10,8), 2, 5], 'Poly_kernel')   
% An example for Random_kernel:
%     w = elm_kernel_dd(a, 0.1, [power(10,8), 50], 'Random_kernel')
%
% Default:  FRACREJ=0.1; KERNEL_TYPE='Random_kernel'; C=10^8; L=1000.
%
% See also: datasets, mapppings, dd_roc.
%
%@article{
%	author = {Q. Leng, H. Qi, J. Miao, W. Zhu and G. Su},
%	title = {One-Class Classification with Extreme Learning Machine},
%	journal = {Mathematical Problems in Engineering},
%	year = {2015}, volume = {2015}, article id = {412957}
%}
% Do some checking
if nargin < 3 || isempty(para), para(1)=power(10,8); para(2)=1000; end;
if nargin < 2 || isempty(fracrej), fracrej=0.1; end;
if nargin < 1 || isempty(a)
    W = prmapping(mfilename,{fracrej, para});
    W = setname(W,'SVM Kernel Data Description');
    return ;
end

if ~ismapping(fracrej)
    
%============================ training ============================ 

	% Make sure a is a OC dataset:
	if ~isocset(a), error('one-class dataset expected'); end
    
    a = +target_class(a);
    [m,k] = size(a);
    svmtype = '-s 2';
    n = para(1);
    g = para(2);
    model = svmtrain(ones(m,1),a,[svmtype,' -n ',num2str(n),' -g ',num2str(g)]);
    
    % find the threshold
    W.threshold = 0;    
    W.model = model;
    W = prmapping(mfilename,'trained',W, char('target','outlier'),k,2);
    W = setname(W,'SVM Data Description');
else
    
%============================ testing ============================
    W = getdata(fracrej);
    [m,k] = size(a);
    model = W.model;
    [~,~,out] = svmpredict(ones(m,1),+a,model);
    newout = [out, repmat(W.threshold,m,1)];
    %new_out
    W = setdat(a,newout,fracrej);
    
end
