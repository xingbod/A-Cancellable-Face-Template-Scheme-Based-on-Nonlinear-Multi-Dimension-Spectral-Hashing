%train_set  train_label gallery gallery_label probe_c probe_o1 probe_o2  probe_o3



train_data.X = insightface_train_set';
sig=-1;
% train_data.S =1-pdist2(train_set, train_set, 'cosine');
% train_data.S =exp(-(1-pdist2(train_set, train_set, 'euclidean'))/(2*sig));
train_data.S =uint8(insightface_train_label==insightface_train_label');
% 25k as test-set
Nb=1024;
opts.lambda = 0.5;% 0.5 1 2
opts.beta = 1;% 0.5 0.8 1
opts.K = 16;
opts.dX = size(insightface_train_set,2);
opts.L = ceil(Nb / ceil(log2(opts.K))); % train maximum number of bits
opts.gaussian=1; %1/0=gaussian/laplace
% original eer


veriFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for face verification ROC plot
osiFarPoints = [0, kron(10.^(-4:-1), 1:9), 1]; % FAR points for open-set face identification ROC plot
rankPoints = [1:10, 20:10:100]; % rank points for open-set face identification CMC plot
reportVeriFar = 0.001; % the FAR point for verification performance reporting
reportOsiFar = 0.01; % the FAR point for open-set identification performance reporting
reportRank = 1; % the rank point for open-set identification performance reporting


numTrials = 1;

numVeriFarPoints = length(veriFarPoints);
iom_random_VR = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
iom_random_veriFAR = zeros(numTrials, numVeriFarPoints); % verification false accept rates of the 10 trials

numOsiFarPoints = length(osiFarPoints);
numRanks = length(rankPoints);
iom_random_DIR = zeros(numRanks, numOsiFarPoints, numTrials); % detection and identification rates of the 10 trials
iom_random_osiFAR = zeros(numTrials, numOsiFarPoints); % open-set identification false accept rates of the 10 trials

%% Get the FAR or rank index where we report performance.
[~, veriFarIndex] = ismember(reportVeriFar, veriFarPoints);
[~, osiFarIndex] = ismember(reportOsiFar, osiFarPoints);
[~, rankIndex] = ismember(reportRank, rankPoints);



%*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&*&&&&&&&&&&&&&&&&&&&&&
% model = learning_IoM(train_data, opts);
model = random_IoM(opts);

db_data.X=insightface_gallery';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_gallery=all_code.Hx';

db_data.X=insightface_probe_c';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_probe_c=all_code.Hx';

db_data.X=insightface_probe_o1';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_probe_o1=all_code.Hx';

db_data.X=insightface_probe_o2';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_probe_o2=all_code.Hx';

db_data.X=insightface_probe_o3';
[all_code, ~] = IoM(db_data, opts, model);
hashed_code_probe_o3=all_code.Hx';


% Compute the cosine similarity score between the test samples.
score_c =1-(pdist2( hashed_code_gallery,hashed_code_probe_c,  'jaccard'));
% Evaluate the verification performance.
[iom_random_VR(1,:), iom_random_veriFAR(1,:)] = EvalROC(score_c, insightface_gallery_label, insightface_probe_label_c, veriFarPoints);

% CMC close set
match_similarity =1-(pdist2( hashed_code_probe_c, hashed_code_gallery, 'jaccard'));
[iom_random_max_rank,iom_random_rec_rates] = CMC(match_similarity,insightface_probe_label_c,insightface_gallery_label);

score_o1 =1-(pdist2( hashed_code_gallery,hashed_code_probe_o1,  'jaccard'));
% Evaluate the open-set identification performance.
[iom_random_DIR(:,:,1), iom_random_osiFAR(1,:)] = OpenSetROC(score_o1, insightface_gallery_label, insightface_probe_label_o1, osiFarPoints );

score_o2 =1-(pdist2( hashed_code_gallery,hashed_code_probe_o2,  'jaccard'));
[iom_random_DIR(:,:,2), iom_random_osiFAR(2,:)] = OpenSetROC(score_o2, insightface_gallery_label, insightface_probe_label_o2, osiFarPoints );


score_o3 =1-(pdist2( hashed_code_gallery,hashed_code_probe_o3,  'jaccard'));
[iom_random_DIR(:,:,3), iom_random_osiFAR(3,:)] = OpenSetROC(score_o3, insightface_gallery_label, insightface_probe_label_o3, osiFarPoints );



save('data/iom_random_veriFAR.mat','iom_random_veriFAR');
save('data/iom_random_max_rank.mat','iom_random_max_rank');
save('data/iom_random_VR.mat','iom_random_VR');
save('data/iom_random_rec_rates.mat','iom_random_rec_rates');

save('data/iom_random_osiFAR.mat','iom_random_osiFAR');
save('data/iom_random_DIR.mat','iom_random_DIR');
