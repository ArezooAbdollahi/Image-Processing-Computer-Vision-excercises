function [distance] = HogMeasure(Im1,Im2,patch)
    vecImg1 = HOGfeature(Im1,patch);
    vecImg2 = HOGfeature(Im2,patch);
    
    minSize = min(size(vecImg1,2), size(vecImg2,2));
    sumdot =0;
    sumI1=0;
    sumI2=0;
    for i=1:minSize
        sumdot = sumdot+vecImg1(1,i).*vecImg2(1,i);
        sumI1 = sumI1 + vecImg1(1,i).^2;
        sumI2 = sumI2 + vecImg2(1,i).^2;
    end
    if (size(vecImg1)>minSize)
        for i=minSize+1:size(vecImg1)
            sumI1 = sumI1 + vecImg1(1,i).^2;
        end
    end
    if (size(vecImg2)>minSize)
        for i=minSize+1:size(vecImg2)
            sumI2 = sumI2 + vecImg2(1,i).^2;
        end
    end
    
    distance = sumdot./(sqrt(sumI1).*sqrt(sumI2));
end