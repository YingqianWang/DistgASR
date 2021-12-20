%% Initialization
clear all;
clc;

%% Parameters setting
angRes_in = 2;
angRes_out = 7;

Type = 'Lytro';

sourceDataPath = ['../Datasets/', Type, '/test/'];
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
        
        data = single(zeros(angRes_in * H, angRes_in * W));
        label = single(zeros(angRes_out * H, angRes_out * W));
        
        for u = 1 : 6 : 7
            for v = 1 : 6 : 7                
                temp = squeeze(LF(u, v, :, :, :));
                temp = rgb2ycbcr(temp);
                u0 = (u-1)/6 + 1;
                v0 = (v-1)/6 + 1;
                data((u0-1)*H+1 : u0*H, (v0-1)*W+1 : v0*W) = squeeze(temp(:,:,1));                
            end
        end
        
        for u = 1 : angRes_out
            for v = 1 : angRes_out
                temp = squeeze(LF(u, v, :, :, :));
                temp = rgb2ycbcr(temp);
                label((u-1)*H+1 : u*H, (v-1)*W+1 : v*W) = squeeze(temp(:,:,1));
            end
        end
        
        SavePath = ['../Data/TestData_', Type, '_2x2-7x7/', DatasetName, '/'];
        if exist(SavePath, 'dir')==0
            mkdir(SavePath);
        end
        
        SavePath_H5 = [SavePath, sceneName, '.h5'];        
        h5create(SavePath_H5, '/data', size(data), 'Datatype', 'single');
        h5write(SavePath_H5, '/data', single(data), [1,1], size(data));
        h5create(SavePath_H5, '/label', size(label), 'Datatype', 'single');
        h5write(SavePath_H5, '/label', single(label), [1,1], size(label));
        
    end
end

