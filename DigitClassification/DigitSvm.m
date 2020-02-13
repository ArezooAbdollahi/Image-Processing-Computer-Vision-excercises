function [approximate] = DigitSvm(cellsize)
    url = 'C:\Program Files\MATLAB\R2014a\toolbox\vision\visiondemos\digits';

    %for Training:
    
    %Read Train's Images & Find Hog Vector of each & save in a Cell Array
    urlTrain = strcat(url,'\handwritten');
    srcAddrs = dir(fullfile(urlTrain));
    for i = 3 : size(srcAddrs,1)
        filename = strcat(urlTrain,'\',srcAddrs(i).name);
        subfile = dir(fullfile(filename));

        for j=3 : size(subfile,1)
            imageName = strcat(filename,'\',subfile(j).name);
            I = imread(imageName);

            %Remove Noises
            Thresh = graythresh(I);
            I = im2bw(I,Thresh);
            
            %save Hog vector in a Cell Array
            imageTrain(i-2,j-2) ={extractHOGFeatures(I,'cellsize',cellsize)};
        end
    end

    hogSize = length(imageTrain{1,1});
    
    %build Data base of Hogs & Lables
    featureVectorTrain = zeros(((size(srcAddrs,1)-2)*(size(subfile,1)-2)),hogSize);
    featureVectorLable = (-1)*ones(((size(srcAddrs,1)-2)*(size(subfile,1)-2)),10);
    
    index=1;
    for i=1:size(imageTrain,1)
        for j=1:size(imageTrain,2)
            featureVectorTrain(index,:)=imageTrain{i,j};
            featureVectorLable(index,i)=i-1;
            index=index+1;
        end
    end

    for i=1:size(imageTrain,1)
        Struct(i) = svmtrain(featureVectorTrain,featureVectorLable(:,i));
    end


    %For Testing:
    
    %Read Test's Images & find Hog Vector & save in a cell Array
    urlTest = strcat(url,'\synthetic');
    srcAddrs = dir(fullfile(urlTest));
    
    for i = 3 : size(srcAddrs,1)
        filename = strcat(urlTest,'\',srcAddrs(i).name);
        subfile = dir(fullfile(filename));

        for j=3 : size(subfile,1)
            imageName = strcat(filename,'\',subfile(j).name);
            I = imread(imageName);
            
            %Remove Noises
            Thresh = graythresh(I);
            I = im2bw(I,Thresh);
            %Save in a cell array
            imageTest(i-2,j-2) ={extractHOGFeatures(I,'cellsize',cellsize)};
        end
    end

    featureVectorTest = zeros(((size(srcAddrs,1)-2)*(size(subfile,1)-2)),hogSize);
    
    %Build Data base of Test's Hog Vector
    index=1;
    for i=1:size(imageTest,1)
        for j=1:size(imageTest,2)
            featureVectorTest(index,:)=imageTest{i,j};
            index=index+1;
        end
    end

    %Labeling
    for i=1:size(imageTest,1)
        group(:,i)=svmclassify(Struct(i),featureVectorTest);
    end

    %For approximate
    approximate = zeros(4,10);
    for i=1:size(group,2)
        Rightanswer = 0;
        total = 0;
        for j=1:size(group,1)
            if(group(j,i)==(i-1))
                total = total+1;
                if ( ((i-1)*(size(subfile,1)-2))<j && j<(i*(size(subfile,1)-2)) )
                    Rightanswer = Rightanswer+1;
                end
            end
        end
        approximate(1,i)=total;
        approximate(2,i)=Rightanswer;
        approximate(3,i)=total-Rightanswer;
        approximate(4,i)=Rightanswer./(size(subfile,1)-2);
    end
end