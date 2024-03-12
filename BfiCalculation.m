function F=BfiCalculation()

    clc, clear, close

%% read test reference & sample image 

    % read reference image
    ref_image=Tiff('ref.tiff','r');
    ref_image_data=read(ref_image);
    imagesc(ref_image_data)    
    colorbar

    [x,y,xr,yr]=SelectRoi('ROI of refernce',101);
    close
    % mean reference intensity 
    mean_Ir=mean2(ref_image_data(int16(y):int16(y+yr),int16(x):int16(x+xr)));
    
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
    [x1,y1,xr1,yr1]=SelectRoi('ROI of refence and sample',101);    
    close
    
    image_fft = fft2(ref_sam_image_data(y1:y1+yr1, x1:x1+xr1));
    %%
    image_fft_shift = fftshift(image_fft);
    imagesc(log(abs(image_fft_shift)))
    colorbar
    [x2,y2,xr2,yr2]=SelectRoi('ROI of interference',11);
    [x3,y3,xr3,yr3]=SelectRoi('ROI of sample',11);
    close

%% fft shift and image plot
    
    cd(MyData)

    for i = 1:len

        num = num2str(i);
        str = strcat("ss_single_",num,".tiff");
        tiff = Tiff(str);
        r = read(tiff);
        image_fft = fft2(r(y1:y1+yr1, x1:x1+xr1));
        image_fft_shift = fftshift(image_fft);
        
        roi_fft = image_fft_shift(y2:y2+yr2, x2:x2+xr2);
        Is = image_fft_shift(y3:y3+yr3, x3:x3+xr3);
        
        Is=ifft2(Is);
        Is=abs(ifftshift(Is));
        mean_Is=mean2(Is);

        roi_fft=ifft2(roi_fft);
        roi_fft=abs(ifftshift(roi_fft));
        roi_fft=mean2(roi_fft);
    
        image_fft_shift_abs = roi_fft;
        FFTimage_kj(:,:,i) = image_fft_shift_abs;

        
        F(i) = mean(image_fft_shift_abs,"all")/(mean_Is*mean_Ir);
    end

    cd(current_folder) 
%%    
    %plot F

    fs=0.1;
    time=fs:fs:len*fs;
    plot(time,F)
    xlabel('time')
    ylabel('F')
    title("F value","FontSize",10)

end



