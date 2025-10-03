clear;
close all;

%% Read and show images
img = imread("data/arducam_img.png");

figure(1);
subplot(1,2,1);
imshow(img);
title('Original image');

subplot(1,2,2);
imhist(img);
ylabel('number of pixels');
title('image intensity histogram');

%% Increase image contrast

img_low_expo = imread("data/arducam_img_low_exposure.png");

figure(2);
subplot(1,2,1);
imshow(uint8(img_low_expo));
title('Low exposured image');

subplot(1,2,2);
imhist(uint8(img_low_expo));
ylabel('number of pixels');
title('image intensity histogram');


img_low_expo = double(img_low_expo);
img_scaled = img_low_expo .* 5;
for i = 1:size(img_scaled, 1)
    for j = 1:size(img_scaled, 2)
        if img_scaled(i,j) > 255
            img_scaled(i,j) = 255;
        end
    end
end

figure(3);
subplot(1,2,1);
imshow(uint8(img_scaled));
title('Image with increased contrast');

subplot(1,2,2);
imhist(uint8(img_scaled));
ylabel('number of pixels');
title('image intensity histogram');

%% Thresholding

img_cam_man = imread('camera_man.png');

figure(4);
subplot(1,2,1);
imshow(img_cam_man);
title('Camera man');

subplot(1,2,2);
imhist(uint8(img_cam_man));
ylabel('number of pixels');
title('image intensity histogram');

%% Binary thresholding
thresh = 70;

b_img_cam_man = double(img_cam_man);
for i = 1:size(b_img_cam_man, 1)
    for j = 1:size(b_img_cam_man, 2)
        if b_img_cam_man(i,j) < 70
            b_img_cam_man(i,j) = 0;
        else
            b_img_cam_man(i,j) = 255;
        end
    end
end

figure(5);
imshow(uint8(b_img_cam_man));
title('Binarized Image');

