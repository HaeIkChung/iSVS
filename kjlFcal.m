%%
clc,close,clear
%%
input = inputdlg({'dscaxleft','dscaxright','dscaydown','dscayup','idscaxleft','idscaxright','idscaydown','idscayup','fft xleft','fft xright','fft ydown','fft yup'},...
              'Customer', [1 50; 1 50; 1 50; 1 50; 1 50; 1 50; 1 50; 1 50;1 50; 1 50; 1 50; 1 50]); 
%%
reft = Tiff("ref785.tiff");
refr = read(reft);
reffft = fft2(refr(1350:1550,1300:1500));

reffftshift = fftshift(reffft);
absfft = abs(reffftshift);
roifrefft = absfft(90:110,90:110);
%%
for i = 1:1800
    num = num2str(i);
    str = strcat("ss_single_",num,".tiff");
    tiff = Tiff(str);
    r = read(tiff);
    imagefft = fft2(r(1350:1550,1300:1500));
    imagefftshift = fftshift(imagefft);
%     absshift = log(abs(imagefftshift));
    
    absshift = abs(imagefftshift);
    Is = absshift(90:110,90:110);
    realIs = mean(Is,"all") - 0.5*10^6;
    R(i) = realIs;
    roifft = absshift(130:140,110:120);
    
    imagefftshiftabs = roifft;
    FFTimage_kj(:,:,i) = imagefftshiftabs;

    F(i) = mean(imagefftshiftabs,"all")/realIs;
%     % dsca
    DSCAimage_kj(:,:,i) = r(1050:1070,900:920);
    B = im2col(DSCAimage_kj(:,:,i),[7 7],'distinct');
    B = double(B);
    MEAN = mean(B);
    STD = std(B,1);
    K = STD./MEAN;
    BFI_box = (1./K.^2)'; % 9x1 column vector 
    % didn't work when 1/K'.^2 was used (figure out later)
    
    BFI_box_mean = mean(BFI_box);
    BFI_box_std = std(BFI_box);
    
    final_BFI_kj(i) = BFI_box_mean;
    final_BFI_std_kj(i)= BFI_box_std;



    

end
%%
subplot(3,1,1)
imagesc(roifrefft)
colorbar()
subplot(3,1,2)
imagesc(absshift(90:110,90:110))
colorbar()
subplot(3,1,3)
imagesc(Is)
colorbar()
%%
subplot(2,1,1)
imagesc(refr(1350:1550,1300:1500))
colorbar()
subplot(2,1,2)
imagesc(r(1350:1550,1300:1500))
colorbar()
%%
refrsum = mean(refr(1350:1550,1300:1500),"all");
rsum = mean(r(1350:1550,1300:1500),"all");
%%
save("FFTimage_kj.mat","FFTimage_kj");
save("DSCAimage_kj,mat","DSCAimage_kj");
%%
ti = (1:1800)/600;
plot(ti,F)
xlabel("time (min)","FontSize",20)
ylabel("F","FontSize",20)
title("F value","FontSize",25)
%%

ti = (1:1800)/600;
plot(ti,R)
xlabel("time (min)")
ylabel("R")
title("R graph")

%%
ti = (1:1800)/600;
plot(ti,final_BFI_kj)
xlabel("time (min)","FontSize",20)
ylabel("BFI","FontSize",20)
title("BFI value","FontSize",25)