% EDA for DDA on ABCD
% Y.C. 03/23/2023
%%
clear; close all
addpath('/home/yuchen/Documents/dCov_nonlinear/')
addpath('/home/yuchen/Documents/fMRI_Real/SupportFunctions/FSLNets/')
addpath('/home/yuchen/Documents/fMRI_Real/SupportFunctions/L1precision/')
addpath('~/Documents/Balloon/Manuscript/SupportFunctions/');
addpath('/home/claudia/TOOLS/')
addpath('~/Documents/MATLAB/cbrewer/cbrewer')
addpath('~/Documents/HCPrediction/')
addpath('~/Documents/HCPrediction/spikeRNN/')
addpath('~/Documents/fMRI_Real/SupportFunctions/2019_03_03_BCT/')
addpath('/home/yuchen/Documents/fMRI_Real/SupportFunctions/powerlaws_full_v0.0.10-2012-01-17') % for power law dist function fitting
cd /cnl/abcd/data/imaging/fmri/rsfmri/interim/

data_dir = '/nadata/cnl/abcd/data/imaging/fmri/rsfmri/interim/segmented/baseline/downloads/';
ddc_dir='/nadata/cnl/abcd/data/imaging/fmri/rsfmri/interim/DDC/baseline/raw/';
TR = 0.8;

subjects = dir(data_dir);
subjects=subjects(3:end);

%%

to_check=[];
tic
for i=1:length(subjects)

    % get all segmented fMRI data (sessions) 
    if ~exist([ddc_dir subjects(i).name], 'dir')
        mkdir([ddc_dir subjects(i).name]);
    end

    cd([ddc_dir subjects(i).name]);

    if ~exist('single_sessions', 'dir')
        mkdir('single_sessions');
    end

    
    fprintf(['Computing DDC on ' subjects(i).name '\n'])
    cd([data_dir subjects(i).name])
    data=dir('filt_cort*');

    for k=1:length(data)
        if isfile(['Reg_DDC2H_' data(k).name(end-9:end)])
            fprintf(['DDC Already computed on ' subjects(i).name  '\n'])
    
        else
        


            % load single session fmri data
            cd([data_dir subjects(i).name])
            ts = dlmread([data(k).name]);

            % DDC dual-hemisphere analysis
            V = ts;
            [T, N] = size(V); % number of timepoints x number of nodes
            V_obs = zscore(V);
            [dCov1,dCov2,~,dCov_center] = dCov_numerical(V_obs,TR);
            [Cov,Precision,B,~] = estimators(V_obs,0,TR);
            C = dCov2;
            B = Cov;
            A_reg = zeros(size(C));
            lambda = 1e-2;
            for n = 1:size(C,1)
                ci = C(n,:);
                % [coef,stats]=lasso(B,transpose(ci),'Lambda',lambda);
                coef = ridge(transpose(ci),B,lambda);
                A_reg(n,:) = transpose(coef);
            end
            
            cd([ddc_dir subjects(i).name '/single_sessions'])
            csvwrite( ['Reg_DDC2H_' data(k).name(end-9:end)] , A_reg )
            %figure, imagesc(A_reg)

        end
            

    
    end

end
toc
