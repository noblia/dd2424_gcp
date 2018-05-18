function [imgs] = getRawData(no_imgs)
% to access the coordinates: img.epithelial.detection
img.epithelial = [];
img.fibroblast = [];
img.inflammatory = [];
img.others = [];
img.self = zeros(500,500,3);
imgs = cell(no_imgs,1);
for i = 1:no_imgs
    num = i;
    folder_name = strcat('img',num2str(num));
     % keep colors
    img.self = imread(strcat(folder_name,'.bmp'));
    img.epithelial = load(strcat(folder_name,'_','epithelial'),'detection');
    img.fibroblast = load(strcat(folder_name,'_','fibroblast'));
    img.inflammatory = load(strcat(folder_name,'_','inflammatory'));
    img.others = load(strcat(folder_name,'_','others'));
    
   
    imgs{i} = img;
end
