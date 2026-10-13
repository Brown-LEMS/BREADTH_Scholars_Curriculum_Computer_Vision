clear;
close all;

%% Read and show images
noisy_img = imread("data/noisy_peppers.png");

figure(1);
imshow(noisy_img);
title('Noisy image');

%% Create a 2D discrete Gaussian kernel

kernel_size = 15;
sigma = 2;
ax = linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size);
[xx, yy] = meshgrid(ax, ax);
kernel = exp(-(xx.^2 + yy.^2) / (2*sigma^2));
gauss_kernel_2d = kernel / sum(kernel(:));

figure(2);
imagesc(gauss_kernel_2d);
colormap('jet');
colorbar;
title('2D Discrete Gaussian Kernel Heatmap');
axis square;

%% Convolve the image with the 2D Gaussian kernel

tic;

noisy_img = double(noisy_img);
denoised_img = conv2(noisy_img, gauss_kernel_2d, 'same');

elapsed_time = toc;
fprintf('2D Gaussian convolving the image took %.5f seconds\n', elapsed_time);

figure(3);
imshow(uint8(denoised_img), []);
title('Gaussian Convolved Image');

%% Use Separable 1D Gaussian Kernels

ax = linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size);
gauss_kernel_1d = exp(-0.5 * (ax / sigma).^2);
gauss_kernel_1d = gauss_kernel_1d / sum(gauss_kernel_1d);

tic;
convolve_rows = conv2(noisy_img, gauss_kernel_1d, 'same');
denoised_img_sep = conv2(convolve_rows, gauss_kernel_1d', 'same');
elapsed_time = toc;

fprintf('Separable Gaussian convolving the image took %.5f seconds\n', elapsed_time);

figure(4);
imshow(uint8(denoised_img_sep));
title('Gaussian Convolved Image using separable Gaussian kernel');
