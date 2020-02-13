function main()
    BlockSize = [3 3];
    CellSize = [6 6];
    struct = learning (BlockSize, CellSize);
    list_img = ['image_00000743.jpg'; 'image_00000826.jpg'; 'image_00000911.jpg'; 'image_00000934.jpg'; 'image_00000948.jpg'; ];
%     list_img = ['image_00000911.jpg'];
    for i=1:size(list_img, 1)
        testImage = imread(list_img(i, :));
        searchPedestrisn(testImage , struct , BlockSize , CellSize );
    end
end

function svmStruct=learning(BlockSize, CellSize)
    try
        load('svmStruct.mat', 'svmStruct');
    catch
        thisDir = 'D:\homework\Image processing\homework 6\new_pedestrian';
        negativeDir = fullfile(thisDir, 'cropped_pedestrian', 'images', 'neg'); %997 items
        positiveDir = fullfile(thisDir, 'cropped_pedestrian', 'images', 'pos'); %2003 items cropped _ 924 items 
        nagativeTrainingSet = imageSet( negativeDir );
        positiveTrainingSet = imageSet( positiveDir );

        %get hog feature size
        sampleImage = read( nagativeTrainingSet(1) , 1 );   
        [ sampleHog , vis ] = extractHOGFeatures(sampleImage(1:128, 1:64), 'BlockSize' , BlockSize , 'CellSize' , CellSize );
        hogFeatureSize = length( sampleHog );

        %extract hog features
        numberOfNegativeItems = 997; %997;
        numberOfPositiveItems = 2003; %2003;

        trainingFeatures = zeros( numberOfNegativeItems+numberOfPositiveItems , hogFeatureSize );
        trainingLabels = zeros( numberOfNegativeItems+numberOfPositiveItems , 1 );
    %     figure;
        for i = 1:numberOfNegativeItems  
            image = read( nagativeTrainingSet(1), i ); 
            % crop image
            image = image(1:128, 1:64, :);
            [img_h, img_w, img_d] = size(image);
            %pre-processing
            image = preprocess(image , img_w , img_h , img_d); 
            %hog       
            [ trainingFeatures( i,: ) , vis ] = extractHOGFeatures(image , 'BlockSize', BlockSize, 'CellSize' , CellSize );      
            trainingLabels( i , 1 ) = -1;
        end
        for i = 1:numberOfPositiveItems 
            peopleDetector = vision.PeopleDetector('UprightPeople_128x64');
            image = read( positiveTrainingSet(1), i );   
            [bboxes , scores] = step(peopleDetector, image);
            size_b = size( bboxes , 1 );
            if size_b ~= 0
                image = image( bboxes(1,2):bboxes(1,2)+bboxes(1,4)-1 , bboxes(1,1):bboxes(1,1)+bboxes(1,3)-1, : );
            end
            image = imresize( image , [128, 64]);
            [img_h , img_w , img_d] = size(image);        
            %pre-processing
            image = preprocess(image, img_w, img_h, img_d); 
            %hog        
            [ trainingFeatures( numberOfNegativeItems+i ,: ) , vis ] = extractHOGFeatures(image , 'BlockSize' , BlockSize , 'CellSize' , CellSize );
            trainingLabels( numberOfNegativeItems+i , 1 ) = 1;
            %imshow(image);
        end

        %normalize hog vector
        %factor = sqrt( trainingFeatures / ( norm(trainingFeatures,1) + 0.04 ) );
        %trainingFeatures = trainingFeatures .* factor;

        %train classifiers
        disp('start learning(svm).')
        svmStruct = libsvmtrain(trainingLabels, trainingFeatures, '-c 1 -g 0.05 -b 1');
        disp('end learning.')
        save('svmStruct.mat', 'svmStruct');
    end
end

function image = preprocess(image, img_w, img_h, img_d)
    %apply guassian
    %filter = fspecial( 'gaussian' , 3 , 0.5 );
    %image = imfilter( image , filter ); 
    %normalize with NRGB
%     if img_d == 3  
%        image = nrgb( image , img_w , img_h , img_d );
%     else
%         image = im2double(image);
%     end
    %convert to grayScale
%     if( img_d > 1 )       
%        image = rgb2gray(image);
%     end
    %convert to double
    image = im2double(image);  
    %apply gradient
    image_dx = imfilter(image , [ -1, 0, 1 ]);
    image_dy = imfilter(image , [ 1; 0; -1 ]);
    image = sqrt(power(image_dx, 2) + power(image_dy, 2));
end

function nrgb=nrgb(img, width, height, depth)
nrgb = zeros(height, width, depth); 
for i = 1:height
    for j =1:width
        nrgb(i, j, :) = double(img(i, j, :)) ./ double(img(i, j, 1) + img(i, j, 2) + img(i, j, 3));
    end
end
end


function searchPedestrisn(image , svmStruct, BlockSize, CellSize)
    [img_h , img_w , img_d ] = size(image);
    
    overlap = 2;
    scaledWindows = [256, 192, 128 ];
    color = ['b'; 'r'; 'y'; 'g'];
    
    large_window_h = max(scaledWindows);
    large_window_w = floor((64/128)*large_window_h);
    
    imgC = zeros(img_h + floor(large_window_h - mod(img_h-large_window_h-1, large_window_h/overlap)), img_w + floor(large_window_w - mod(img_w-large_window_w-1, large_window_w/overlap)), img_d, 'uint8');
    imgC(1:img_h, 1:img_w, :) = image(:, :, :);
    [img_h, img_w, img_d] = size(imgC);
    
    % walk on image
    list_points = [];
    for i = 1:large_window_h/overlap:img_h - large_window_h + 1
        for j = 1:large_window_w/overlap:img_w - large_window_w + 1
            subImg = imgC(i:i+large_window_h, j:j+large_window_w, :);
            subImg = imresize(subImg, [128, 64]);
            subImg = preprocess(subImg, large_window_h, large_window_w, img_d);
            [features, visualization] = extractHOGFeatures(subImg, 'BlockSize', BlockSize, 'CellSize', CellSize);
            
            [l, a, p] = libsvmpredict(1.0, double(features), svmStruct, '-b 1');
            if p(1, 1) > 0.85 % grater than %40
                list_points = [list_points; [j, i, large_window_w, large_window_h]];
                image = insertText(image,[j, i],num2str(p(1, 1),'%.2f'),'FontSize',18,'BoxColor','r','BoxOpacity',0.4);
            end
        end
    end
    figure, imshow(image), hold on;
    listRectangle(list_points);
    hold off;
end


function listRectangle(points)
    for k=1:size(points, 1)
        rectangle('Position', [points(k, 1), points(k, 2), points(k, 3), points(k, 4)], 'EdgeColor', 'b');
    end
end