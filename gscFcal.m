%%
clc,close,clear
%%
input = inputdlg({'dscaxleft','dscaxright','dscaydown','dscayup','idscaxleft','idscaxright','idscaydown','idscayup','fft xleft','fft xright','fft ydown','fft yup'},...
              'Customer', [1 50; 1 50; 1 50; 1 50; 1 50; 1 50; 1 50; 1 50;1 50; 1 50; 1 50; 1 50]); 
%%
for i = 1:1800
    num = num2str(i);
    str = strcat("ss_single_",num,".tiff");
    tiff = Tiff(str);
    r = read(tiff);
    imagefft = fft2(r(1350:1550,1300:1500));
    imagefftshift = fftshift(imagefft);
    absshift = log(abs(imagefftshift));
    roifft = imagefft(125:145,105:125);
    
    Is = absshift(90:110,90:110);
    realIs = mean(Is,"all") - 0.5*10^6;
    R(i) = realIs;
    roifft = absshift(130:140,110:120);
    
    imagefftshiftabs = roifft;
    FFTimage_gs(:,:,i) = imagefftshiftabs;

    F(i) = mean(imagefftshiftabs,"all")/realIs;;

    F(i) = mean(imagefftshiftabs,"all");
    % dsca
    DSCAimage_gs(:,:,i) = r(1050:1070,900:920);

    B = im2col(DSCAimage_gs(:,:,i),[7 7],'distinct');
    B = double(B);
    MEAN = mean(B);
    STD = std(B,1);
    K = STD./MEAN;
    BFI_box = (1./K.^2)'; % 9x1 column vector 
    % didn't work when 1/K'.^2 was used (figure out later)
    
    BFI_box_mean = mean(BFI_box);
    BFI_box_std = std(BFI_box);
    
    final_BFI_gs(i) = BFI_box_mean;
    final_BFI_std_gs(i)= BFI_box_std;



    

end
%%
save("FFTimage_gs.mat","FFTimage_gs");
save("DSCAimage_gs,mat","DSCAimage_gs");
%%
ti = (1:1800)/600;
plot(ti,F)
xlabel("time (min)")
ylabel("BFI(A.U.)")
title("gsc F value")
%%
ti = (1:1800)/600;
plot(ti,final_BFI_gs)
xlabel("time (min)")
ylabel("BFI")
title("gsc BFI value")