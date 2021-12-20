%% Initialization
clear all;
clc;

%% Parameters setting
angRes_in = 2;
angRes_out = 7;

Type = 'Lytro';

sourceDataPath = ['../Dataset/', Type, '/test/'];
sourceDatasets = dir(sourceDataPath);
sourceDatasets(1:2) = [];
datasetsNum = length(sourceDatasets);
idx = 0;

%% Test data generation
for DatasetIndex = 1 : datasetsNum
    sourceDataFolder = [sourceDataPath, sourceDatasets(DatasetIndex).name, '/'];
    DatasetName = sourceDatasets(DatasetIndex).name;
    folders = dir(sourceDataFolder);
    folders(1:2) = [];
    sceneNum = length(folders);
    
    for iScene = 1 : sceneNum
        idx_s = 0;
        sceneName = folders(iScene).name;
        sceneName(end-3:end) = [];
        fprintf('Generating test data of Scene_%s in Dataset %s......\t\n', sceneName, sourceDatasets(DatasetIndex).name);
        dataPath = [sourceDataFolder, folders(iScene).name];
        data = load(dataPath);
        
        LF = data.LF;
        [U, V, H, W, ~] = size(LF);
        
        LF = LF(2:8, 2:8, :, :, 1:3);        
        
        idx = idx + 1;
        idx_s = idx_s + 1;
        
        SavePath = ['./input/', Type, '/', DatasetName, '_', sceneName, '/'];
        if exist(SavePath, 'dir')==0
            mkdir(SavePath);
        end
        for u = 1 : 6 : 7
            for v = 1 : 6 : 7                
                temp = squeeze(LF(u, v, :, :, :));
                imwrite(temp, [SavePath, 'view_', num2str(u, '%02d'), '_', num2str(v, '%02d'), '.png']);                               
            end
        end       
      
        
    end
end

