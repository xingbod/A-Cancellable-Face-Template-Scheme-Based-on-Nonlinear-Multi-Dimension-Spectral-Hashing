% addpath('/matlab_tools');
% addpath_recurse("/matlab_tools")
addpath('MDSH');



SHparamNew.nbits = 1024; % number of bits to code each sample
SHparamNew.doPCA=0;
SHparamNew1 = trainMDSH(insightface_train_set, SHparamNew);

save('data/SHparamNew1.mat','SHparamNew1');

 
% original eer


veriFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for face verification ROC plot
osiFarPoints = [0, kron(10.^(-4:-1), 1:9), 1]; % FAR points for open-set face identification ROC plot
rankPoints = [1:10, 20:10:100]; % rank points for open-set face identification CMC plot
reportVeriFar = 0.001; % the FAR point for verification performance reporting
reportOsiFar = 0.01; % the FAR point for open-set identification performance reporting
reportRank = 1; % the rank point for open-set identification performance reporting


numTrials = 1;

numVeriFarPoints = length(veriFarPoints);
iom_VR = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
iom_veriFAR = zeros(numTrials, numVeriFarPoints); % verification false accept rates of the 10 trials

numOsiFarPoints = length(osiFarPoints);
numRanks = length(rankPoints);
iom_DIR = zeros(numRanks, numOsiFarPoints, numTrials); % detection and identification rates of the 10 trials
iom_osiFAR = zeros(numTrials, numOsiFarPoints); % open-set identification false accept rates of the 10 trials

%% Get the FAR or rank index where we report performance.
[~, veriFarIndex] = ismember(reportVeriFar, veriFarPoints);
[~, osiFarIndex] = ismember(reportOsiFar, osiFarPoints);
[~, rankIndex] = ismember(reportRank, rankPoints);


SHparamNew1.softmod=0; % set with 0.6 as non-linear mdsh, introduce non-linearity
%*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&
 % model = random_IoM(opts);


randnum=orth(rand(size(insightface_gallery,2)));

for a=1:size(insightface_gallery,1)
    new_insightface_gallery(a,:)=insightface_gallery(a,:)* randnum;
end

for a=1:size(insightface_probe_c,1)
    new_insightface_probe_c(a,:)=insightface_probe_c(a,:)* randnum;
end

for a=1:size(insightface_probe_o1,1)
    new_insightface_probe_o1(a,:)=insightface_probe_o1(a,:)* randnum;
end

for a=1:size(insightface_probe_o2,1)
    new_insightface_probe_o2(a,:)=insightface_probe_o2(a,:)* randnum;
end

for a=1:size(insightface_probe_o3,1)
    new_insightface_probe_o3(a,:)=insightface_probe_o3(a,:)* randnum;
end


[B1,U1] = compressMDSH(new_insightface_gallery, SHparamNew1);
U1 = sign(U1);
hashed_code_gallery=U1;

[B1,U1] = compressMDSH(new_insightface_probe_c, SHparamNew1);
U1 = sign(U1);
hashed_code_probe_c=U1;

[B1,U1] = compressMDSH(new_insightface_probe_o1, SHparamNew1);
U1 = sign(U1);
hashed_code_probe_o1=U1;

[B1,U1] = compressMDSH(new_insightface_probe_o2, SHparamNew1);
U1 = sign(U1);
hashed_code_probe_o2=U1;

[B1,U1] = compressMDSH(new_insightface_probe_o3, SHparamNew1);
U1 = sign(U1);
hashed_code_probe_o3=U1;


% Compute the cosine similarity score between the test samples.
score_c =1-(pdist2( hashed_code_gallery,hashed_code_probe_c,  'jaccard'));
% Evaluate the verification performance.
[iom_VR(1,:), iom_veriFAR(1,:)] = EvalROC(score_c, insightface_gallery_label, insightface_probe_label_c, veriFarPoints);

% CMC close set
match_similarity =1-(pdist2( hashed_code_probe_c, hashed_code_gallery, 'jaccard'));
[iom_max_rank,iom_rec_rates] = CMC(match_similarity,insightface_probe_label_c,insightface_gallery_label);

score_o1 =1-(pdist2( hashed_code_gallery,hashed_code_probe_o1,  'jaccard'));
% Evaluate the open-set identification performance.
[iom_DIR(:,:,1), iom_osiFAR(1,:)] = OpenSetROC(score_o1, insightface_gallery_label, insightface_probe_label_o1, osiFarPoints );

score_o2 =1-(pdist2( hashed_code_gallery,hashed_code_probe_o2,  'jaccard'));
[iom_DIR(:,:,2), iom_osiFAR(2,:)] = OpenSetROC(score_o2, insightface_gallery_label, insightface_probe_label_o2, osiFarPoints );


score_o3 =1-(pdist2( hashed_code_gallery,hashed_code_probe_o3,  'jaccard'));
[iom_DIR(:,:,3), iom_osiFAR(3,:)] = OpenSetROC(score_o3, insightface_gallery_label, insightface_probe_label_o3, osiFarPoints );



save('data/iom_veriFAR.mat','iom_veriFAR');
save('data/iom_max_rank.mat','iom_max_rank');
save('data/iom_VR.mat','iom_VR');
save('data/iom_rec_rates.mat','iom_rec_rates');

save('data/iom_osiFAR.mat','iom_osiFAR');
save('data/iom_DIR.mat','iom_DIR');
