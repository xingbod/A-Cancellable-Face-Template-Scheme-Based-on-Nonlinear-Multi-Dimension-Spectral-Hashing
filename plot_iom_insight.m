

veriFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for face verification ROC plot
osiFarPoints = [0, kron(10.^(-4:-1), 1:9), 1]; % FAR points for open-set face identification ROC plot
rankPoints = [1:10, 20:10:100]; % rank points for open-set face identification CMC plot
reportVeriFar = 0.001; % the FAR point for verification performance reporting
reportOsiFar = 0.01; % the FAR point for open-set identification performance reporting
reportRank = 1; % the rank point for open-set identification performance reporting

%% Get the FAR or rank index where we report performance.
[~, veriFarIndex] = ismember(reportVeriFar, veriFarPoints);
[~, osiFarIndex] = ismember(reportOsiFar, osiFarPoints);
[~, rankIndex] = ismember(reportRank, rankPoints);

for count=1:5
    
     
    load(['data/deep_insight_veriFAR',num2str(count),'.mat'],'deep_insight_veriFAR');
    load(['data/deep_insight_max_rank',num2str(count),'.mat'],'deep_insight_max_rank');
    load(['data/deep_insight_VR',num2str(count),'.mat'],'deep_insight_VR');
    load(['data/deep_insight_rec_rates',num2str(count),'.mat'],'deep_insight_rec_rates');
    load(['data/deep_insight_osiFAR',num2str(count),'.mat'],'deep_insight_osiFAR');
    load(['data/deep_insight_DIR',num2str(count),'.mat'],'deep_insight_DIR');
        
    load(['data/insight_iom_random_veriFAR',num2str(count),'.mat'],'iom_random_veriFAR');
    load(['data/insight_iom_random_max_rank',num2str(count),'.mat'],'iom_random_max_rank');
    load(['data/insight_iom_random_VR',num2str(count),'.mat'],'iom_random_VR');
    load(['data/insight_iom_random_rec_rates',num2str(count),'.mat'],'iom_random_rec_rates');
    load(['data/insight_iom_random_osiFAR',num2str(count),'.mat'],'iom_random_osiFAR');
    load(['data/insight_iom_random_DIR',num2str(count),'.mat'],'iom_random_DIR');
 
    load(['data/insight_iom_veriFAR',num2str(count),'.mat'],'iom_veriFAR');
    load(['data/insight_iom_max_rank',num2str(count),'.mat'],'iom_max_rank');
    load(['data/insight_iom_VR',num2str(count),'.mat'],'iom_VR');
    load(['data/insight_iom_rec_rates',num2str(count),'.mat'],'iom_rec_rates');
    load(['data/insight_iom_osiFAR',num2str(count),'.mat'],'iom_osiFAR');
    load(['data/insight_iom_DIR',num2str(count),'.mat'],'iom_DIR');
    
    
    load(['data/insight_biohashing_veriFAR',num2str(count),'.mat'],'biohashing_veriFAR');
    load(['data/insight_biohashing_max_rank',num2str(count),'.mat'],'biohashing_max_rank');
    load(['data/insight_biohashing_VR',num2str(count),'.mat'],'biohashing_VR');
    load(['data/insight_biohashing_rec_rates',num2str(count),'.mat'],'biohashing_rec_rates');
    load(['data/insight_biohashing_osiFAR',num2str(count),'.mat'],'biohashing_osiFAR');
    load(['data/insight_biohashing_DIR',num2str(count),'.mat'],'biohashing_DIR');
    
    
     % feature_fusion_VR  feature_fusion_rec_rates feature_fusion_DIR
    
    new_deep_insight_VR(count,:)=deep_insight_VR(1,:);
    new_iom_random_VR(count,:)=iom_random_VR(1,:);
    new_iom_VR(count,:)=iom_VR(1,:);
    new_biohashing_VR(count,:)=biohashing_VR(1,:);
     
    
    new_deep_insight_rec_rates(count,:)=deep_insight_rec_rates;
    new_iom_random_rec_rates(count,:)=iom_random_rec_rates;
    new_iom_rec_rates(count,:)=iom_rec_rates;
    new_biohashing_rec_rates(count,:)=biohashing_rec_rates;
 
    new_deep_insight_DIR1(count,:)=deep_insight_DIR(rankIndex,:,1);
    new_deep_insight_DIR2(count,:)=deep_insight_DIR(rankIndex,:,2);
    new_deep_insight_DIR3(count,:)=deep_insight_DIR(rankIndex,:,3);
    
    new_iom_random_DIR1(count,:)=iom_random_DIR(rankIndex,:,1);
    new_iom_random_DIR2(count,:)=iom_random_DIR(rankIndex,:,2);
    new_iom_random_DIR3(count,:)=iom_random_DIR(rankIndex,:,3);
    
    new_iom_DIR1(count,:)=iom_DIR(rankIndex,:,1);
    new_iom_DIR2(count,:)=iom_DIR(rankIndex,:,2);
    new_iom_DIR3(count,:)=iom_DIR(rankIndex,:,3);
     
    
    new_biohashing_DIR1(count,:)=biohashing_DIR(rankIndex,:,1);
    new_biohashing_DIR2(count,:)=biohashing_DIR(rankIndex,:,2);
    new_biohashing_DIR3(count,:)=biohashing_DIR(rankIndex,:,3);
     
    
    
end


close all;
figure(1)
%% Plot the face verification ROC curve.
semilogx(deep_insight_veriFAR(1,:) * 100, mean(new_deep_insight_VR)* 100,'-+', 'LineWidth', 1);
hold on;
semilogx(iom_random_veriFAR(1,:) * 100, mean(new_iom_random_VR)* 100,'-x', 'LineWidth', 1);
hold on;
semilogx(biohashing_veriFAR(1,:) * 100, mean(new_biohashing_VR)* 100,'-s', 'LineWidth', 1);
hold on;
semilogx(iom_veriFAR(1,:) * 100,  mean(new_iom_VR)* 100,'-^', 'LineWidth', 1);
xlim([0,100]); ylim([0,100]); grid on;
xlabel('False Accept Rate (%)');
ylabel('Verification Rate (%)');
title('Close-set C Face Verification ROC Curve');
legend(["Insight-Feature","Index-of-Max Hashing(IoM)","Biohashing","MDSH"],'location','southeast')
saveas(gcf,['graph\insight_norm_deep_c_roc.tif']);

mean(new_deep_insight_VR(:,[29 38 56]))* 100
mean(new_iom_random_VR(:,[29 38 56]))* 100
mean(new_iom_VR(:,[29 38 56]))* 100

figure(2)
semilogx(1:deep_insight_max_rank, mean(new_deep_insight_rec_rates)* 100,'-+', 'LineWidth', 1);
hold on
semilogx(1:iom_random_max_rank, mean(new_iom_random_rec_rates)* 100,'-x', 'LineWidth', 1);
hold on
semilogx(1:biohashing_max_rank, mean(new_biohashing_rec_rates)* 100,'-s', 'LineWidth', 1);
hold on
semilogx(1:iom_max_rank, mean(new_iom_rec_rates)* 100,'-^', 'LineWidth', 1);
xlim([0,deep_insight_max_rank]); ylim([99,100]); grid on;
xlabel('Rank');
ylabel('Identification Rate');
title('Close-set C Face Verification CMC Curve');
legend(["Insight-Feature","Index-of-Max Hashing(IoM)","Biohashing","MDSH"],'location','southeast')

x_formatstring = '%6.1f';
% Here's the code.
xtick = get(gca, 'xtick');
for i = 1:length(xtick)
    xticklabel{i} = sprintf(x_formatstring, xtick(i));
end
set(gca, 'xticklabel', xticklabel);
saveas(gcf,['graph\insight_norm_deep_c_cmc.tif']);
 mean(new_deep_insight_rec_rates(:,1))* 100
 mean(new_iom_random_rec_rates(:,1))* 100
 mean(new_iom_rec_rates(:,1))* 100

figure(3)
%% Plot the open-set face Identification DIR Curve at the report rank.
semilogx(deep_insight_osiFAR(1,:)* 100, mean(new_deep_insight_DIR1) * 100,'-+', 'LineWidth', 1);
hold on
semilogx(iom_random_osiFAR(1,:)* 100, mean(new_iom_random_DIR1) * 100,'-x', 'LineWidth', 1);
hold on
semilogx(biohashing_osiFAR(1,:)* 100, mean(new_biohashing_DIR1) * 100,'-s', 'LineWidth', 1);
hold on
semilogx(iom_osiFAR(1,:)* 100, mean(new_iom_DIR1) * 100,'-^', 'LineWidth', 1);
xlim([0,100]); ylim([0,100]); grid on;
xlabel('False Alarm Rate (%)');
ylabel('Detection and Identification Rate (%)');
title(sprintf('Open-set O1 Identification DIR Curve at Rank %d', reportRank));
legend(["Insight-Feature","Index-of-Max Hashing(IoM)","Biohashing","MDSH"],'location','southeast')
saveas(gcf,['graph\insight_norm_deep_o1_roc.tif']);

% mean(new_deep_insight_DIR3(:,[40 56])) * 100
% mean(new_iom_random_DIR3(:,[4 20])) * 100
% mean(new_iom_DIR3(:,[4 20])) * 100
%  
figure(4)
%% Plot the open-set face Identification DIR Curve at the report rank.
semilogx(deep_insight_osiFAR(2,:)* 100, mean(new_deep_insight_DIR2) * 100,'-+', 'LineWidth', 1);
hold on
semilogx(iom_random_osiFAR(2,:)* 100, mean(new_iom_random_DIR2)  * 100,'-x', 'LineWidth', 1);
hold on
semilogx(biohashing_osiFAR(2,:)* 100, mean(new_biohashing_DIR2) * 100,'-s', 'LineWidth', 1);
hold on
semilogx(iom_osiFAR(2,:)* 100, mean(new_iom_DIR2)  * 100,'-^', 'LineWidth', 1);
xlim([0,100]); ylim([0,100]); grid on;
xlabel('False Alarm Rate (%)');
ylabel('Detection and Identification Rate (%)');
title(sprintf('Open-set O2 Identification DIR Curve at Rank %d', reportRank));
legend(["Insight-Feature","Index-of-Max Hashing(IoM)","MDSH"],'location','southeast')
saveas(gcf,['graph\insight_norm_deep_o2_roc.tif']);

figure(5)
%% Plot the open-set face Identification DIR Curve at the report rank.
semilogx(deep_insight_osiFAR(3,:)* 100,  mean(new_deep_insight_DIR3) * 100,'-+', 'LineWidth', 1);
hold on
semilogx(iom_random_osiFAR(3,:)* 100,  mean(new_iom_random_DIR3) * 100,'-x', 'LineWidth', 1);
hold on
semilogx(biohashing_osiFAR(3,:)* 100, mean(new_biohashing_DIR3) * 100,'-s', 'LineWidth', 1);
hold on
semilogx(iom_osiFAR(3,:)* 100,  mean(new_iom_DIR3) * 100,'-^', 'LineWidth', 1);
xlim([0,100]); ylim([0,100]); grid on;
xlabel('False Alarm Rate (%)');
ylabel('Detection and Identification Rate (%)');
title(sprintf('Open-set O3 Identification DIR Curve at Rank %d', reportRank));
legend(["Insight-Feature","Index-of-Max Hashing(IoM)","Biohashing","MDSH"],'location','southeast')
saveas(gcf,['graph\insight_norm_deep_o3_roc.tif']);


%deep_facenet_veriFAR(1,:) * 100 %VR@FAR=0.001%	VR@FAR=0.01%	VR@FAR=1% 0.001-29 0.01-38 0.1-47 1-56 

%deep_facenet_osiFAR(1,:)* 100 47 56  deep_insight_osiFAR(1,:)* 100 deep_fusion_osiFAR(1,:)* 100
% mean(new_deep_insight_DIR3(:,47)) * 100
% mean(new_iom_random_DIR3(:,47)) * 100
% mean(new_iom_DIR3(:,47)) * 100
