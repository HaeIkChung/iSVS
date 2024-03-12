function ImageRead()
%% read test reference & sample image 
    
    clc, close, clear

    % read reference image
    subplot(2,3,1)
    ref_image=Tiff('ref.tiff','r');
    ref_image_data=read(ref_image);
    imagesc(ref_image_data)
    colorbar
    title("Reference","FontSize",10)
    
    %read sample image
    subplot(2,3,2)
    sample_image=Tiff('sam.tiff','r');
    sample_image_data=read(sample_image);
    imagesc(sample_image_data)
    colorbar
    title("sample","FontSize",10)

    %read reference & sample image
    subplot(2,3,3)
    ref_sample_image=Tiff('ref&sam.tiff','r');
    ref_sample_image_data=read(ref_sample_image);
    imagesc(ref_sample_image_data)
        
    roi = drawrectangle('Position',[700,700,300,300],'StripeColor','y');
    pause
    x=roi.Position(1);
    y=roi.Position(2);
    r1=roi.Position(3);
    r2=roi.Position(4);
    colorbar
    title("ref&sample","FontSize",10)
    
    %fft ref & sample image 
    subplot(2,3,4)
    test = fft2(ref_sample_image_data(int16(x):int16(x+r1),int16(y):int16(y+r2)));
    testshift = fftshift(test);
    absshift = log(abs(testshift));
    imagesc(absshift)
    
    colorbar
    axis image
    title("2D FFT result abuout Reference + smaple beam","FontSize",10)

end