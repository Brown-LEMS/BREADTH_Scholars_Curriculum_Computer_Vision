

img = imread("data/arducam_img.png");

figure(1);
imshow(img);

figure(2);
imhist(img);
ylabel('number of pixels');
title('image intensity histogram');

img = double(img);
img = img .* 2.2;
for i = 1:size(img, 1)
    for j = 1:size(img, 2)
        if img(i,j) > 255
            img(i,j) = 255;
        end
    end
end

figure(3);
imshow(uint8(img));

figure(4);
imhist(uint8(img));
ylabel('number of pixels');
title('image intensity histogram');

