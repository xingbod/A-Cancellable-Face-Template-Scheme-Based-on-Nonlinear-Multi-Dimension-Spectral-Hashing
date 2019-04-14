clear all;
close all;
addpath('matlab_tools');
addpath_recurse("matlab_tools")
% 
 


for count=1:5
    generateSet_Insight
    deepFeatures_insight
    iom_random_insight
    mdsh_insight
    biohashing_insight
    save(['data/deep_insight_veriFAR',num2str(count),'.mat'],'deep_insight_veriFAR');
    save(['data/deep_insight_max_rank',num2str(count),'.mat'],'deep_insight_max_rank');
    save(['data/deep_insight_VR',num2str(count),'.mat'],'deep_insight_VR');
    save(['data/deep_insight_rec_rates',num2str(count),'.mat'],'deep_insight_rec_rates');
    save(['data/deep_insight_osiFAR',num2str(count),'.mat'],'deep_insight_osiFAR');
    save(['data/deep_insight_DIR',num2str(count),'.mat'],'deep_insight_DIR');
        
    save(['data/insight_iom_random_veriFAR',num2str(count),'.mat'],'iom_random_veriFAR');
    save(['data/insight_iom_random_max_rank',num2str(count),'.mat'],'iom_random_max_rank');
    save(['data/insight_iom_random_VR',num2str(count),'.mat'],'iom_random_VR');
    save(['data/insight_iom_random_rec_rates',num2str(count),'.mat'],'iom_random_rec_rates');
    save(['data/insight_iom_random_osiFAR',num2str(count),'.mat'],'iom_random_osiFAR');
    save(['data/insight_iom_random_DIR',num2str(count),'.mat'],'iom_random_DIR');
 
    save(['data/insight_iom_veriFAR',num2str(count),'.mat'],'iom_veriFAR');
    save(['data/insight_iom_max_rank',num2str(count),'.mat'],'iom_max_rank');
    save(['data/insight_iom_VR',num2str(count),'.mat'],'iom_VR');
    save(['data/insight_iom_rec_rates',num2str(count),'.mat'],'iom_rec_rates');
    save(['data/insight_iom_osiFAR',num2str(count),'.mat'],'iom_osiFAR');
    save(['data/insight_iom_DIR',num2str(count),'.mat'],'iom_DIR');
    
    
    
    save(['data/insight_biohashing_veriFAR',num2str(count),'.mat'],'biohashing_veriFAR');
    save(['data/insight_biohashing_max_rank',num2str(count),'.mat'],'biohashing_max_rank');
    save(['data/insight_biohashing_VR',num2str(count),'.mat'],'biohashing_VR');
    save(['data/insight_biohashing_rec_rates',num2str(count),'.mat'],'biohashing_rec_rates');
    save(['data/insight_biohashing_osiFAR',num2str(count),'.mat'],'biohashing_osiFAR');
    save(['data/insight_biohashing_DIR',num2str(count),'.mat'],'biohashing_DIR');
    
    
end

plot_iom_insight
