function [x,y,r1,r2]=SelectRoi(title,range)
    %select ROI 
    roi = drawpoint("Deletable",true,"Label",title,"Color",'r');
    x=roi.Position(1);
    y=roi.Position(2);
    pause
    delete(roi)

    roi = drawrectangle('Position',[x-((range-1)/2),y-((range-1)/2),range,range],'Label',title,'StripeColor','y');
    x=round(roi.Position(1));
    y=round(roi.Position(2));
    r1=round(roi.Position(3));
    r2=round(roi.Position(4));
    pause    
end