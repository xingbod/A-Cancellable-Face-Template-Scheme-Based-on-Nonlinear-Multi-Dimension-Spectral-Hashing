clear all
close all;
addpath('matlab_tools');
addpath_recurse("matlab_tools")
addpath('../MDSH');
load('E:\my research\EURECOM2018\EURECOM-Thermal Face\data\lfw\LFW_label_10Samples_insightface.mat')
load('E:\my research\EURECOM2018\EURECOM-Thermal Face\data\lfw\LFW_10Samples_insightface.mat')
% 

randnum=orth(rand(size(LFW_10Samples_insightface,2)));

for a=1:size(LFW_10Samples_insightface,1)
    new_LFW_10Samples_insightface(a,:)=LFW_10Samples_insightface(a,:)* randnum;
end

SHparamNew.nbits = 1024; % number of bits to code each sample
SHparamNew.doPCA=0;
SHparamNew1 = trainMDSH(new_LFW_10Samples_insightface(randperm(1580,400),:), SHparamNew);
SHparamNew1.softmod=1;
SHparamNew1.alpha=0.5;
[B1,U1] = compressMDSH(new_LFW_10Samples_insightface, SHparamNew1);
U1 = sign(U1);
hashed_code_gallery=U1;


distance=1-pdist2( hashed_code_gallery,hashed_code_gallery,  'jaccard');
gen_score = distance(LFW_label_10Samples_insightface'==LFW_label_10Samples_insightface);
imp_score = distance(LFW_label_10Samples_insightface'~=LFW_label_10Samples_insightface);
[EER, mTSR, mFAR, mFRR, mGAR] =computeperformance(gen_score, imp_score, 0.001);  % isnightface 3.43 % 4.40 %
% plothisf(gen_score,imp_score,'bit',1,1,2000000);



randnum2=orth(rand(size(LFW_10Samples_insightface,2)));

for a=1:size(LFW_10Samples_insightface,1)
    new_LFW_10Samples_insightface(a,:)=LFW_10Samples_insightface(a,:)* randnum2;
end

SHparamNew.nbits = 1024; % number of bits to code each sample
SHparamNew.doPCA=0;
SHparamNew1 = trainMDSH(new_LFW_10Samples_insightface(randperm(1580,400),:), SHparamNew);
SHparamNew1.softmod=1;
SHparamNew1.alpha=0.5;

[B1,U1] = compressMDSH(new_LFW_10Samples_insightface, SHparamNew1);
U1 = sign(U1);
hashed_code_gallery2=U1;


distance=1-pdist2( hashed_code_gallery,hashed_code_gallery2,  'jaccard');
mated_gen_score = distance(LFW_label_10Samples_insightface'==LFW_label_10Samples_insightface);
nonmated_imp_score = distance(LFW_label_10Samples_insightface'~=LFW_label_10Samples_insightface);
[EER2, mTSR, mFAR, mFRR, mGAR] =computeperformance(mated_gen_score, nonmated_imp_score, 0.001);  % isnightface 3.43 % 4.40 %
% plothisf(gen_score,imp_score,'bit',1,1,2000000);
gen_score = gen_score(find(gen_score~=1)); % exclude same sample match
% plothisf_revocable(gen_score,imp_score(randperm(length(imp_score),length(gen_score))),mated_gen_score,'bit',1,1,6000);
plothisf_unlinkability(mated_gen_score,nonmated_imp_score(randperm(length(nonmated_imp_score),length(mated_gen_score))),'bit',1,1,6000);
saveas(gcf,['graph\unlinkability.tif']);


