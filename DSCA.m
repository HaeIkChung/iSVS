function BFI=DSCA()
    
    clc, clear, close

    %find location of images
    current_folder=pwd;
    cd ..\
    MyData=uigetdir();
    cd(MyData)

    temp=dir('*.tiff');
    len=size(temp,1);

    ref_image=Tiff('ss_single_1.tiff','r');
    ref_image_data=read(ref_image);
    imagesc(ref_image_data)
    colorbar
    cd(current_folder)
    [x,y,xr,yr]=SelectRoi('ROI of refence',101);  
    close

    BFI=zeros(len);
    
    cd(MyData)
%%
    for i = 1:len

        num = num2str(i);
        str = strcat("ss_single_",num,".tiff");
        tiff = Tiff(str);
        r=read(tiff);
        temp = r(y:y+yr, x:x+xr);

        %calculate K
        MEAN = mean2(temp);
        STD = std2(temp);
        K = STD./MEAN;
        BFI(i) = 1/K;
        
    end

    cd(current_folder) 
%%
    fs=0.1;
    time=fs:fs:len*fs;
    plot(time,BFI)
    hold on
    xline(60)
    hold on
    xline(120)
    xlabel('time')
    ylabel('BFI')
    title("BFI-DSCA","FontSize",10)
end