function [VecImg] = HOGfeature (Im,patch)
    Img = Im;
    x = size(Img,1);
    y = size(Img,2);
    
    %Gussian Filter
    Gus = 1./16 * [1,2,1;2,4,2;1,2,1]; 
    Img = uint8(conv2(Img,Gus,'same'));
    
    %Plot results
    figure;
    imshow(Im);title('real pic');
    figure;
    imshow(Img);title('after gussian filtering');
    
    
    % Make Differential's Matrix 
    Dx=[-1 0 1];
    Dy=[1;0;-1];
    
    Ix = conv2(Img,Dx,'same');
    Iy = conv2(Img,Dy,'same');
    
    %Plot results
    figure;
    subplot(1,2,1);imshow(Ix);title('Ix');
    subplot(1,2,2);imshow(Iy);title('Iy');
    
    %find gradient & theta
    g = uint8( sqrt(double(Ix.^2)+double(Iy.^2)));
    %Plot results
    figure; imshow(g); title('G');
    
    %initialize theta
    theta = zeros(x,y);
    
    for i=1:x
        for j=1:y
            theta(i,j) = (atan2(Iy(i,j),Ix(i,j)))*180/pi;
            if (theta(i,j)<0)
                theta(i,j)=theta(i,j)+360;
            end 
        end
    end
    
    %for find vector & colorsMap
    bin = zeros(9,5);
    bin(1,1)=0;
    bin(1,2)=40;
    bin(1,3)=255;
    bin(1,4)=0;
    bin(1,5)=0;
    
    bin(2,1)=40;
    bin(2,2)=80;
    bin(2,3)=255;
    bin(2,4)=128;
    bin(2,5)=0;
    
    bin(3,1)=80;
    bin(3,2)=120;
    bin(3,3)=255;
    bin(3,4)=255;
    bin(3,5)=0;
    
    bin(4,1)=120;
    bin(4,2)=160;
    bin(4,3)=0;
    bin(4,4)=255;
    bin(4,5)=0;
    
    bin(5,1)=160;
    bin(5,2)=200;
    bin(5,3)=0;
    bin(5,4)=255;
    bin(5,5)=128;
    
    bin(6,1)=200;
    bin(6,2)=240;
    bin(6,3)=0;
    bin(6,4)=255;
    bin(6,5)=255;
    
    bin(7,1)=240;
    bin(7,2)=280;
    bin(7,3)=0;
    bin(7,4)=0;
    bin(7,5)=255;
    
    bin(8,1)=280;
    bin(8,2)=320;
    bin(8,3)=128;
    bin(8,4)=0;
    bin(8,5)=255;
    
    bin(9,1)=320;
    bin(9,2)=360;
    bin(9,3)=255;
    bin(9,4)=0;
    bin(9,5)=255;
    
    %Plot a empty matrix (make a black background)
    result = zeros(x,y,3);
    
    %vector for every bin
    vector=zeros(1,9);
    %final vector for Image
    sizeOfVec=ceil(x./patch)*ceil(y./patch);
    VecImg = zeros(1,sizeOfVec);
    endSize=1;
    
    %find vectors
    for i=1:patch:x
        for j=1:patch:y

            for k=1:patch
                if (i+k > x) %last row,when there is no enough pixel
                    break;
                end
                for h=1:patch
                    if (j+h > y) %last column,when there is no enough pixel
                        break;
                    end
                    
                    for l1=1:9
                        if ((theta(i+(k-1),j+(h-1))>=bin(l1,1)) && (theta(i+(k-1),j+(h-1))<bin(l1,2)))
                            vector(l1) = vector(l1)+1;
                        end
                    end
                end
            end
            
            %normaling vector
            normvec=0;
            for l1=1:9
                normvec = normvec+ vector(l1).^2;
            end
            normvec = sqrt(double(normvec));
                        
            maxbin=0;
            numMaxbin=0;

            for l1=1:9
                if (vector(l1)>=maxbin)        
                    maxbin=vector(l1);
                    numMaxbin=l1;
                end
                vector(l1)=vector(l1)./normvec;
                VecImg(1,endSize) = vector(l1);
                endSize = endSize+1;
            end
            
            %coloring a bin with maximum color
            for num1=1:k
                for num2=1:h
                    result(i+(num1-1),j+(num2-1),1)=bin(numMaxbin,3);
                    result(i+(num1-1),j+(num2-1),2)=bin(numMaxbin,4);
                    result(i+(num1-1),j+(num2-1),3)=bin(numMaxbin,5);
                end
            end
        end
    end
    
    %plot result
    figure ; imshow(result);
end