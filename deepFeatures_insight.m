close all;
addpath('matlab_tools');
addpath_recurse("matlab_tools")
%train_set  train_label gallery gallery_label probe_c probe_o1 probe_o2  probe_o3

veriFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for face verification ROC plot
osiFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for open-set face identification ROC plot
rankPoints = [1:10, 20:10:100]; % rank points for open-set face identification CMC plot
reportVeriFar = 0.001; % the FAR point for verification performance reporting
reportOsiFar = 0.001; % the FAR point for open-set identification performance reporting
reportRank = 1; % the rank point for open-set identification performance reporting


numTrials = 1;

numVeriFarPoints = length(veriFarPoints);
deep_insight_VR = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
deep_insight_veriFAR = zeros(numTrials, numVeriFarPoints); % verification false accept rates of the 10 trials

numOsiFarPoints = length(osiFarPoints);
numRanks = length(rankPoints);
deep_insight_DIR = zeros(numRanks, numOsiFarPoints, numTrials); % detection and identification rates of the 10 trials
deep_insight_osiFAR = zeros(numTrials, numOsiFarPoints); % open-set identification false accept rates of the 10 trials

%% Get the FAR or rank index where we report performance.
[~, veriFarIndex] = ismember(reportVeriFar, veriFarPoints);
[~, osiFarIndex] = ismember(reportOsiFar, osiFarPoints);
[~, rankIndex] = ismember(reportRank, rankPoints);


% Compute the cosine similarity score between the test samples.
score_c =1./(pdist2( insightface_gallery,insightface_probe_c,  'euclidean')+0.001);

% Evaluate the verification performance.
[deep_insight_VR(1,:), deep_insight_veriFAR(1,:)] = EvalROC(score_c, insightface_gallery_label, insightface_probe_label_c, veriFarPoints);

% CMC close set
match_similarity =1./(pdist2( insightface_probe_c, insightface_gallery, 'euclidean')+0.001);
[deep_insight_max_rank,deep_insight_rec_rates] = CMC(match_similarity,insightface_probe_label_c,insightface_gallery_label);

score_o1 =1./(pdist2( insightface_gallery,insightface_probe_o1,  'euclidean')+0.001);
% Evaluate the open-set identification performance.
[deep_insight_DIR(:,:,1), deep_insight_osiFAR(1,:)] = OpenSetROC(score_o1, insightface_gallery_label, insightface_probe_label_o1, osiFarPoints,rankPoints );

score_o2 =1./(pdist2( insightface_gallery,insightface_probe_o2,  'euclidean')+0.001);
[deep_insight_DIR(:,:,2), deep_insight_osiFAR(2,:)] = OpenSetROC(score_o2, insightface_gallery_label, insightface_probe_label_o2, osiFarPoints ,rankPoints);


score_o3 =1./(pdist2( insightface_gallery,insightface_probe_o3,  'euclidean')+0.001);
[deep_insight_DIR(:,:,3), deep_insight_osiFAR(3,:)] = OpenSetROC(score_o3, insightface_gallery_label, insightface_probe_label_o3, osiFarPoints ,rankPoints);


save('data/deep_insight_veriFAR.mat','deep_insight_veriFAR');
save('data/deep_insight_max_rank.mat','deep_insight_max_rank');
save('data/deep_insight_VR.mat','deep_insight_VR');
save('data/deep_insight_rec_rates.mat','deep_insight_rec_rates');
save('data/deep_insight_osiFAR.mat','deep_insight_osiFAR');
save('data/deep_insight_DIR.mat','deep_insight_DIR');
% 