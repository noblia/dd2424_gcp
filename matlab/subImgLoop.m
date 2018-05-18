function imgs = subImgLoop(img,cell_list, sub_img_size, curr_img)

no_cells = size(cell_list,1);
imgs = zeros(27, 27,3, no_cells);
for i = 1:no_cells
    curr_img
    imgs(:,:,:, i) = getSubImage(img, cell_list(i,:),sub_img_size);
end
