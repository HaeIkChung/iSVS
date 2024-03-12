%%
clc,close,clear
%%
subplot(2,3,1)
refsam785t = Tiff("ref&sample.tiff");
refsam785 = read(refsam785t);
imagesc(refsam785)
colorbar()
axis image
title("Reference + smaple beam","FontSize",20)
subplot(2,3,2)
ref785t = Tiff("ref.tiff");
ref785 = read(ref785t);
imagesc(ref785)
colorbar()
axis image
title("Reference beam","FontSize",20)

subplot(2,3,3)
sam785t = Tiff("sample.tiff");
sam785 = read(sam785t);
imagesc(sam785)
colorbar()
axis image
title("Sample beam","FontSize",20)

subplot(2,3,4)
test = fft2(refsam785(1350:1550,1300:1500));
testshift = fftshift(test);
absshift = log(abs(testshift));
imagesc(absshift)
colorbar()
axis image
title("2D FFT result abuout Reference + smaple beam","FontSize",20)

subplot(2,3,5)
test2 = fft2(ref785(1350:1550,1300:1500));
testshift2 = fftshift(test2);
absshift2 = log(abs(testshift2));
imagesc(absshift2)
colorbar()
axis image
title("2D FFT result abuout Reference beam","FontSize",20)
subplot(2,3,6)
test3 = fft2(sam785(1350:1550,1300:1500));
testshift3 = fftshift(test3);
absshift3 = log(abs(testshift3));
imagesc(absshift3)
colorbar()
axis image
title("2D FFT result abuout smaple beam","FontSize",20)
%%
t = Tiff("ss_single_1.tiff");
rt = read(t);
subplot(2,1,1)
imagesc(rt)
colorbar()
axis image
subplot(2,1,2)
test = fft2(rt(1320:1530,1320:1530));
testshift = fftshift(test);
absshift = log(abs(testshift));
% maxabs = max(absshift,[],"all");
% imagesc(absshift/maxabs)
imagesc(absshift)
colorbar()
axis image