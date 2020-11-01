clc,clear
addpath('dd_tools')
addpath('prtools')

load('A_1.mat');
target = target_class(A,1);
nuRange = fliplr([0.01,0.05,0.1]);
Srange = power(10,-4:1:4);


%% Self-adaptive Data Shifting (SDS)
ed_t = 0.1;
in_t = 0.1;
N=1;
p = 0.1;
coef = 1;
[all_outlier,edgeset,new_target] = adaptive_syn_gen_final(target, ed_t, coef);
[w1,best_para] = ocsvm_validation_with_target(target, gendatoc(new_target,all_outlier), 'libsvm_kernel_dd', [], {nuRange,Srange}, 'RBF_kernel');



%% plotting
hold on
plot(+target(:,1),+target(:,2),'b.','MarkerSize',17.5) % plot the target data
target = +target;
plotc(w1,'r-',2) % plot the OCC enclosing surface
axis([-8 10 -10 8]) % A_1
% axis([-0.5 4 -0.5 4]) % A_2
% axis([0 12 0 4]) % A_3
% axis([-0 1 -0 1])% A_4
% axis([-5.5 5 -5.5 5])
% axis([-70 65 -70 65]) % A_8
% axis([-8 8 -8 8]) % A_11
% axis([-7 7 -7 7]) % A_17
