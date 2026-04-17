patternDims = [7 10];
family = "DICT_4X4_1000";
imageSize = [900 1200];
checkerSize = 100;
markerSize = 75;

I = generateCharucoBoard(imageSize,patternDims,family,checkerSize,markerSize);

figure;
imshow(I);