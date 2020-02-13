% function [approximate] = DigitClassification(cellSize)
cellSize=[2 2];
% Load training and test data using imageSet.
syntheticDir   = fullfile(toolboxdir('vision'), 'visiondata','digits','synthetic');
handwrittenDir = fullfile(toolboxdir('vision'), 'visiondata','digits','handwritten');

% imageSet recursively scans the directory tree containing the images.
trainingImages = imageSet(syntheticDir,   'recursive');
testImages     = imageSet(handwrittenDir, 'recursive');


 
SFeatures = [];
HFeatures = [];
StrainingFeatures=[];
HtestingFeatures=[];
%for Training:

for d=1:10
    
    numtrainingImages = trainingImages(d).Count;
    % Extract HOG features from each training image.
    trainingLabels = ones(numtrainingImages, 1) * (d - 1);
    for i = 1:numtrainingImages
        img = read(trainingImages(d),i);
        %remove noise
        lvl = graythresh(img);
        img = im2bw(img, lvl);
        StrainingFeatures(i,:) = extractHOGFeatures(img,'CellSize',cellSize);
    end
   
    %build Data base of Hogs
    SFeatures = [SFeatures; StrainingFeatures];
        
end

%build Data base of labels

for d=1:10
    
    Slabels=ones(1010, 1)* (-1);
    start=(d-1)*101;
    Slabels(start+1:start+101)=d-1; 
    Struct(d) = svmtrain(SFeatures,Slabels);

end

%for testing

for d=1:10
    numtrainingImages = testImages(d).Count;
    % Extract HOG features from each testing image
    for i = 1:numtrainingImages
        img = read(testImages(d),i);
        %remove noise
        lvl = graythresh(img);
        img = im2bw(img, lvl);
        HtestingFeatures(i,:) = extractHOGFeatures(img,'CellSize',cellSize);
    end
    %build Data base of Hogs
    HFeatures = [HFeatures; HtestingFeatures];
        
end

%build Data base of labels

for d=1:10
   
    group(:,d) = svmclassify(Struct(d),HFeatures);

end

    %For approximate
    approximate = zeros(4,10);
    for i=1:size(group,2)
        Rightanswer = 0;
        total = 0;
        for j=1:size(group,1) 
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
