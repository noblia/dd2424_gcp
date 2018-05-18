function [images, labels] = dataLoop(cells, sub_img_size)
images = [];
labels = [];
for i = 1:size(cells)
    % 0 = epithelial
    data_epithelial = subImgLoop(cells{i}.self, cells{i}.epithelial.detection, sub_img_size,i);
    labels_epithelial=  zeros(size(cells{i}.epithelial.detection,1),1);
    
    % 1 = fibroblast
    data_fibroblast= subImgLoop(cells{i}.self, cells{i}.fibroblast.detection, sub_img_size,i);
    labels_fibroblast = ones(size(cells{i}.fibroblast.detection,1),1);
    
    % 2 = inflammatory
    data_inflammatory= subImgLoop(cells{i}.self, cells{i}.inflammatory.detection, sub_img_size,i);
    labels_inflammatory =  2*ones(size(cells{i}.inflammatory.detection,1),1);
    
    % 3 = others
    data_others= subImgLoop(cells{i}.self, cells{i}.others.detection, sub_img_size,i);
    labels_others= 3*ones(size(cells{i}.others.detection,1),1);
     
    %collect all info for this cell
    labels = cat(1,labels,labels_epithelial,labels_fibroblast, labels_inflammatory, labels_others);
    images = cat(4, images, data_epithelial, data_fibroblast, data_inflammatory, data_others);
    
          
end