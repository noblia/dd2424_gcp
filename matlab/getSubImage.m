function [subimage] = getSubImage(img, center, sub_img_size)
% returns a smaller image from a bigger image
% img is the original image
% center is the center coordinate of subimage
% size is the image length of subimage

border = size(img,1);
R = imref2d(size(img));
[i,j] = worldToSubscript(R, center(1), center(2));
if isnan(i) || isnan(j)
    subimage = zeros(sub_img_size,sub_img_size,3);
else
    [offset1, offset2] = getOffset(i, sub_img_size, border);
    [offset3, offset4] = getOffset(j, sub_img_size, border);
    subimage = img(i-offset1:i+offset2, j-offset3: j+offset4,:);
    
end


