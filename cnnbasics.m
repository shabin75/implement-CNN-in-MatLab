clc
clear all
close all
path='dataset1'
% data_path=fullfile(path,'CNN','dataset1')
data=imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames')

layers=[imageInputLayer([28 28 1])
    convolution2dLayer(5,20)
    reluLayer
    maxPooling2dLayer(2,'stride',2)
    convolution2dLayer(5,20)
    reluLayer
    maxPooling2dLayer(2,'stride',2)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer()]

options=trainingOptions('sgdm','MaxEpochs',15,'initialLearnRate',0.001)
trainedNet = trainNetwork(data,layers,options)

img=imread('test1.png')

output=classify(trainedNet,img)




