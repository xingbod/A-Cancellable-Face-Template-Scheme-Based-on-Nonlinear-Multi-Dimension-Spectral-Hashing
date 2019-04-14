addpath('matlab_tools');
addpath_recurse("matlab_tools")



% original eer


veriFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for face verification ROC plot
osiFarPoints = [0, kron(10.^(-4:-1), 1:9), 1]; % FAR points for open-set face identification ROC plot
rankPoints = [1:10, 20:10:100]; % rank points for open-set face identification CMC plot
reportVeriFar = 0.001; % the FAR point for verification performance reporting
reportOsiFar = 0.01; % the FAR point for open-set identification performance reporting
reportRank = 1; % the rank point for open-set identification performance reporting


numTrials = 1;

numVeriFarPoints = length(veriFarPoints);
biohashing_VR = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
biohashing_veriFAR = zeros(numTrials, numVeriFarPoints); % verification false accept rates of the 10 trials

numOsiFarPoints = length(osiFarPoints);
numRanks = length(rankPoints);
biohashing_DIR = zeros(numRanks, numOsiFarPoints, numTrials); % detection and identification rates of the 10 trials
biohashing_osiFAR = zeros(numTrials, numOsiFarPoints); % open-set identification false accept rates of the 10 trials

%% Get the FAR or rank index where we report performance.
[~, veriFarIndex] = ismember(reportVeriFar, veriFarPoints);
[~, osiFarIndex] = ismember(reportOsiFar, osiFarPoints);
[~, rankIndex] = ismember(reportRank, rankPoints);



key=orth(rand(size(insightface_train_set,2),1024));



%*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&
 % model = random_IoM(opts);

hashed_code_gallery=double(insightface_gallery*key>0);
hashed_code_probe_c=double(insightface_probe_c*key>0);
hashed_code_probe_o1=double(insightface_probe_o1*key>0);
hashed_code_probe_o2=double(insightface_probe_o2*key>0);
hashed_code_probe_o3=double(insightface_probe_o3*key>0);

% Compute the cosine similarity score between the test samples.
score_c =1-(pdist2( hashed_code_gallery,hashed_code_probe_c,  'jaccard'));
% Evaluate the verification performance.
[biohashing_VR(1,:), biohashing_veriFAR(1,:)] = EvalROC(score_c, insightface_gallery_label, insightface_probe_label_c, veriFarPoints);

% CMC close set
match_similarity =1-(pdist2( hashed_code_probe_c, hashed_code_gallery, 'jaccard'));
[biohashing_max_rank,biohashing_rec_rates] = CMC(match_similarity,insightface_probe_label_c,insightface_gallery_label);

score_o1 =1-(pdist2( hashed_code_gallery,hashed_code_probe_o1,  'jaccard'));
% Evaluate the open-set identification performance.
[biohashing_DIR(:,:,1), biohashing_osiFAR(1,:)] = OpenSetROC(score_o1, insightface_gallery_label, insightface_probe_label_o1, osiFarPoints );

score_o2 =1-(pdist2( hashed_code_gallery,hashed_code_probe_o2,  'jaccard'));
[biohashing_DIR(:,:,2), biohashing_osiFAR(2,:)] = OpenSetROC(score_o2, insightface_gallery_label, insightface_probe_label_o2, osiFarPoints );


score_o3 =1-(pdist2( hashed_code_gallery,hashed_code_probe_o3,  'jaccard'));
[biohashing_DIR(:,:,3), biohashing_osiFAR(3,:)] = OpenSetROC(score_o3, insightface_gallery_label, insightface_probe_label_o3, osiFarPoints );



save('data/biohashing_veriFAR.mat','biohashing_veriFAR');
save('data/biohashing_max_rank.mat','biohashing_max_rank');
save('data/biohashing_VR.mat','biohashing_VR');
save('data/biohashing_rec_rates.mat','biohashing_rec_rates');

save('data/biohashing_osiFAR.mat','biohashing_osiFAR');
save('data/biohashing_DIR.mat','biohashing_DIR');
