function ImageRead()
    
    %% set ROI of image and FFT image
   
    %find location of images
    current_folder=pwd;
    cd ..\
    MyData=uigetdir();
    cd(MyData)

    %find number of images
    temp=dir('*.tiff');
    len=size(temp,1);
    F=zeros(1,len);
    
    %select ROI 
    ref_sam_image=Tiff('ss_single_1.tiff','r');
    ref_sam_image_data=read(ref_sam_image);
    imagesc(ref_sam_image_data)
    colorbar
    cd(current_folder)
    [x1,y1,xr1,yr1]=SelectRoi('ROI of refenc.e and sample',101);    
    close
    
    image_fft = fft2(ref_sam_image_data(y1:y1+yr1, x1:x1+xr1));
    %%
    image_fft_shift = fftshift(image_fft);
    imagesc(log(abs(image_fft_shift)))
    colorbar
    [x2,y2,xr2,yr2]=SelectRoi('ROI of interference',11);
    [x3,y3,xr3,yr3]=SelectRoi('ROI of sample',11);
    close

end