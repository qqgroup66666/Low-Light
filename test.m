clear; clc;
addpath(genpath('/data1/wenqiren/wenqiren/caffe-dilate/'));
caffe.reset_all();

model_path = 'model_G/';
Solver = modelconfig_G_fortest(model_path);

% read the hazy image
lowlight_name = './inputs/10.png';
inputimg = im2single(imread(lowlight_name));   % 512*768
[row, col, cha] = size(inputimg);

% The model can process an image with max size of 320000 pixels on a K80. 
inputimg = imresize(inputimg, [768, 768]);

[gx, gy] = gradient(rgb2gray(inputimg));

batch(:,:,1:3,1) = inputimg;
batch(:,:,4,1) = gx;
batch(:,:,5,1) = gy; 
batchc = {batch};
%Solver.Solver_.net.blobs('data').reshape([height, width, 5, 1]);

% enhance the input using the trained model
tic
activec = Solver.Solver_.net.forward(batchc);
toc

active = activec{1};
output = imresize(active, [row, col]);
imwrite(output, strcat('./results/', lowlight_name(10:end-4),'_enhanced.png'));
