%% Initialization
clear all;
clc;

%% Parameters setting
patchsize = 64;
stride = 16;
angRes_in = 2;
angRes_out = 7;
%Type = 'Lytro';  % HCI or Lytro
sourceDataPath = '../Datasets/HCI/train/';
SavePath = '../Data/TrainData_HCI_2x2-7x7/';

folders = dir(sourceDataPath);
folders(1:2) = [];
sceneNum = length(folders);
idx = 0;

if exist(SavePath, 'dir')==0
    mkdir(SavePath);
end

%% Training data generation
    
    for iScene = 1 : sceneNum
        idx_s = 0;
        sceneName = folders(iScene).name;
        sceneName(end-3:end) = [];
        fprintf('Generating training data of scene %s\t\t', sceneName);
        dataPath = [sourceDataPath, folders(iScene).name];
        data = load(dataPath);
        
        LF = data.LF;
        [U, V, H, W, ~] = size(LF);        
        LF = LF(2:8, 2:8, :, :, 1:3);        
        margin = ceil(stride*rand()) + 1;
        
        for h =  margin : stride : H -  patchsize + 1
            for w =  margin : stride : W -  patchsize + 1
                
                idx = idx + 1;
                idx_s = idx_s + 1;
                
                data = single(zeros(angRes_in * patchsize, angRes_in * patchsize));
                label = single(zeros(angRes_out * patchsize, angRes_out * patchsize));
                               
                for u = 1 : 6 : angRes_out
                    for v = 1 : 6 : angRes_out                        
                        temp = squeeze(LF(u, v, h : h+patchsize-1, w : w+patchsize-1, :));
                        temp = rgb2ycbcr(temp);
                        u0 = (u-1)/6 + 1;
                        v0 = (v-1)/6 + 1;
                        data((u0-1)*patchsize+1 : u0*patchsize, (v0-1)*patchsize+1 : v0*patchsize) = squeeze(temp(:,:,1));                        
                    end
                end
                
                for u = 1 : angRes_out
                    for v = 1 : angRes_out                       
                        temp = squeeze(LF(u, v, h : h+patchsize-1, w : w+patchsize-1, :));
                        temp = rgb2ycbcr(temp);
                        label((u-1)*patchsize+1 : u*patchsize, (v-1)*patchsize+1 : v*patchsize) = squeeze(temp(:,:,1));                        
                    end
                end 
                
                SavePath_H5 = [SavePath, num2str(idx,'%06d'),'.h5'];
                h5create(SavePath_H5, '/data', size(data), 'Datatype', 'single');
                h5write(SavePath_H5, '/data', single(data), [1,1], size(data));
                h5create(SavePath_H5, '/label', size(label), 'Datatype', 'single');
                h5write(SavePath_H5, '/label', single(label), [1,1], size(label));
                
            end
        end
        fprintf([num2str(idx), ' training samples have been generated\n']);
    end


