addpath('../matlab_tools');
addpath_recurse("../matlab_tools")
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

mated_imposter_score=[];
myimposter=[imp_score];
for indx=1:10
    randnum2=orth(rand(size(LFW_10Samples_insightface,2)));
    for a=1:size(LFW_10Samples_insightface,1)
        new_LFW_10Samples_insightface(a,:)=LFW_10Samples_insightface(a,:)* randnum2;
    end
    
    SHparamNew.nbits = 1024; % number of bits to code each sample
    SHparamNew.doPCA=0;
    SHparamNew1 = trainMDSH(new_LFW_10Samples_insightface(randperm(1580,400),:), SHparamNew);
    SHparamNew1.softmod=0;

    [B1,U1] = compressMDSH(new_LFW_10Samples_insightface, SHparamNew1);
    U1 = sign(U1);
    hashed_code_gallery2=U1;
    
    
    distance=1-pdist2( hashed_code_gallery,hashed_code_gallery2,  'jaccard');
    mygen_score = distance(LFW_label_10Samples_insightface'==LFW_label_10Samples_insightface);
    mated_imposter_score = [mated_imposter_score mygen_score ];
        
     distance=1-pdist2( hashed_code_gallery2,hashed_code_gallery2,  'jaccard');
    myimp_score = distance(LFW_label_10Samples_insightface'~=LFW_label_10Samples_insightface);
    myimposter =[myimposter myimp_score];
end

%  
gen_score = gen_score(find(gen_score~=1)); % exclude same sample match
mated_imposter_score1=reshape(mated_imposter_score,1,numel(mated_imposter_score));
imp_score1=reshape(myimposter,1,numel(myimposter));
plothisf_revocable(gen_score,imp_score1(randperm(length(imp_score1),length(mated_imposter_score))),mated_imposter_score1(randperm(length(mated_imposter_score1),length(mated_imposter_score))),'bit',1,1,6000);
saveas(gcf,['graph\revocablity.tif']);
