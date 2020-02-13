function []=DigitClassiffication2(cellsize)
SyntheticDir=fullfile(toolboxdir('vision'),'visiondata','digits','synthetic');
HandwrittenDir=fullfile(toolboxdir('vision'),'visiondata','digits','handwritten');

TrainingSet=imageSet(SyntheticDir,'recursive');
TestingSet=imageSet(HandwrittenDir,'recursive');

trainingFeatures=[];
trainingFeatures=[];
trainingLabels   = [];
for d=1:10
    NumTrainingSet=TrainingSet(d).Count;
     
     
    for i=1:NumTrainingSet
        img=read(TrainingSet(d),i);
        
        lvl=graythresh(img);
        img=im2bw(img,lvl);
        [hog_cellsize, vis_cellsize]=extractHOGFeatures(img,'cellsize',cellsize);
        hogFeatureSize = length(hog_cellsize);
        Features  = zeros(NumTrainingSet, hogFeatureSize, 'single');
        Features(i,:)=extractHOGFeatures(img,'cellsize',cellsize);
    end
    labels = repmat(TrainingSet(d).Description, NumTrainingSet, 1);
    trainingFeatures = [trainingFeatures; Features];   
    trainingLabels   = [trainingLabels; labels];
    
end

classifier = fitcecoc(trainingFeatures, trainingLabels);
[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(TestingSet, hogFeatureSize, cellsize);
predictedLabels = predict(classifier, testFeatures);
confMat = confusionmat(testLabels, predictedLabels);

helperDisplayConfusionMatrix(confMat);