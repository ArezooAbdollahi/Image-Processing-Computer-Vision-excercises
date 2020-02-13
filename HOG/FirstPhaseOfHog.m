function [] = FirstPhaseOfHog (Im,patch,Overlap)
    
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
    Dx = [-1,-2,-1;0,0,0;1,2,1];
    Dy = [-1,0,1;-2,0,2;-1,0,1];
    
    Ix = conv2(Img,Dx,'same');
    Iy = conv2(Img,Dy,'same');
    
    %Plot results
    figure;
    subplot(1,2,1);imshow(Ix);title('Ix');
    subplot(1,2,2);imshow(Iy);title('Iy');
    
    %find gradient & theta
    g = uint8( sqrt(double(Ix.^2)+double(Iy.^2)));
    theta = atan2(Iy,Ix);
    
    %Plot results
    figure; imshow(g); title('G');
    
    %for counting gradient's average magnitude
    g = double(g);
    
    %Plot a empty matrix (make a black background)
    result = zeros(x,y);
    figure ; imshow(result);
    
    %for converting real length to the limited length by Patches
     converting = 255./(sqrt(2.0)*patch./2);
    
    %find avarage of theta & gradient, plot vectors
    for i=1:Overlap:x
        for j=1:Overlap:y
            sumGradient=0.0;
            sumtheta =0;
            for k=1:patch
                if (i+k > x) %last row,when there is no enough pixel
                    break;
                end
                for h=1:patch
                    if (j+h > y) %last column,when there is no enough pixel
                        break;
                    end
                    sumGradient = sumGradient + g(i+(k-1),j+(h-1));
                    sumtheta = sumtheta + theta(i+(k-1),j+(h-1));
                end
            end
            
            magnitude = (sumGradient./(k*h))./converting;
            avgtheta = sumtheta./(k*h);
            v = magnitude * sin (avgtheta);
            u = magnitude * cos (avgtheta);
            xx= i+k/2;
            yy= j+h/2;
            if (magnitude >= (sqrt(2)*patch/8))
                hold on;
                quiver(yy,xx,u,v);
                hold off;
            end
        end
    end
end