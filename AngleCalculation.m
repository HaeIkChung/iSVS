function angle=AngleCalculation()
        
    %find location of images
    current_folder=pwd;
    cd ..\
    MyData=uigetdir();
    cd(MyData)
    
    %select ROI 
    ref_sam_image=Tiff('ss_single_1.tiff','r');
    ref_sam_image_data=read(ref_sam_image);
    imagesc(ref_sam_image_data)
    colorbar
    cd(current_folder)
    [x1,y1,xr1,yr1]=SelectRoi('ROI of refence and sample');    
    close
    
    imagefft = fft2(ref_sam_image_data(int16(x1):int16(x1+xr1),int16(y1):int16(y1+yr1)));
    imagefftshift = fftshift(imagefft);
    imagesc(log(abs(imagefftshift)))
    colorbar
    [x2,y2,xr2,yr2]=SelectRoi('ROI of interference');
    [x3,y3,xr3,yr3]=SelectRoi('ROI of sample');
    close    
    
    %%
    
    

    angle

end