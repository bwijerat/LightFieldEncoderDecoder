
clear all; close all; clc;

%% 

folder = 'E:\bwijerat\Documents\Lnx\EECS6400\LF-intrinsics-data\1';
files = dir(folder);
subFolders = files([files.isdir]);
subFolderNames = [];

views = 9; hw = 512; ch = 3; k_ = 0;
LF = zeros([views, views, hw, hw, ch], 'uint8');

for k = 1 : length(subFolders)
	if not(strcmp(subFolders(k).name, '.') || strcmp(subFolders(k).name, '..') || strcmp(subFolders(k).name, 'OUTPUT_hdf5_1') || strcmp(subFolders(k).name, 'OUTPUT_yuv_1') || strcmp(subFolders(k).name, 'OUTPUT_yuv_2'))
    
        f1 = fullfile(folder, subFolders(k).name);
        fileList = dir(fullfile(f1, '*.png'));
        
        i = 1; j = 1;
        for l = 1 : length(fileList)
            if not(strcmp(fileList(l).name, 'mask.png') || strcmp(fileList(l).name, 'objectids_highres.png') || strcmp(fileList(l).name, 'unused_blenderender_output.png'))

                LF(i, j, :, : ,:) = imread(fullfile(f1, fileList(l).name));

                if j == views
                    i = i + 1; j = 1;
                else
                    j = j + 1;
                end
            end
        end
        
        fprintf('%d) %s: ', k - k_, subFolders(k).name);
        img_list = squeeze(spiral_order(LF, views, views));

        file_out = fullfile(folder, 'OUTPUT_yuv_1', strcat(subFolders(k).name, '.yuv'));        
        fid = fopen(file_out, 'w');
        if (fid < 0) 
            error('Could not open the file!');
        end
        
        for i = 1 : size(img_list, 1)
            img_yuv = rgb2ycbcr(squeeze(img_list(i, :, :, :)));
            
            Y = uint8(img_yuv(:,:,1));
            U = uint8(imresize(img_yuv(:,:,2), 0.5));
            V = uint8(imresize(img_yuv(:,:,3), 0.5));            
            
            fwrite(fid,Y','uint8');     
            fwrite(fid,U','uint8');
            fwrite(fid,V','uint8');   
        end
        fclose(fid);
        fprintf(' Done.\n');
        
    else
        k_ = k_ + 1;
    end
end





