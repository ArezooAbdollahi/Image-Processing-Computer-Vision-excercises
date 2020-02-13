function []=DigitClassiffication3(cellsize)
SyntheticDir=fullfile(toolboxdir('vision'),'visiondata','digits','synthetic');
HandwrittenDir=fullfile(toolboxdir('vision'),'visiondata','digits','handwritten');

TrainingSet=imageSet(SyntheticDir,'recursive');
TestingSet=imageSet(HandwrittenDir,'recursive');

trainingFeatures=[];
trainingLabels= [];
testingFeatures=[];
TFeatures=[];
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
    trainingFeatures = [trainingFeatures; Features];   
    
end
%  SVMModel = fitcsvm(X,Y,'KernelFunction','rbf','Standardize',true,'ClassNames',{'negClass','posClass'});

for d=1:10
    Slabels=ones(1010, 1)* (-1);
    start=(d-1)*101;
    Slabels(start+1:start+101)=d-1;
    
    SVMModel{d} = fitcsvm(trainingFeatures, Slabels,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
%        SVMModel{d} = fitcsvm(trainingFeatures, Slabels);

end

%for testing

for d=1:10
    NumTestingSet=TestingSet(d).Count;
    for i=1:NumTestingSet
        img=read(TestingSet(d),i);
        
        lvl=graythresh(img);
        img=im2bw(img,lvl);
        [hog_cellsize, vis_cellsize]=extractHOGFeatures(img,'cellsize',cellsize);
        hogFeatureSize = length(hog_cellsize);
        TFeatures  = zeros(NumTestingSet, hogFeatureSize, 'single');
        TFeatures(i,:)=extractHOGFeatures(img,'cellsize',cellsize);
    end
    testingFeatures = [testingFeatures; TFeatures];   
end

for d=1:10
  
    [~,score] = predict( SVMModel{d},testingFeatures);
    ScoreSVMModel{d} = fitPosterior(SVMModel{d},trainingFeatures, Slabels)
end

   approximate = zeros(4,10);
    for i=1:size(score(:,1))
        Rightanswer = 0;
        total = 0;
        for j=1:size(score(:,2)) 
            if(group(j,i)==(i-1))
                total = total+1;
                if ( ((i-1)*12)<j && j<(i*12) )
                    Rightanswer = Rightanswer+1;
                end
            end
        end
        approximate(1,i)=total;
        approximate(2,i)=Rightanswer;
        approximate(3,i)=total-Rightanswer;
        approximate(4,i)=Rightanswer./10;
    end


