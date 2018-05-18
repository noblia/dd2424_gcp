function [dataset, labels, raw_data] = getData(no_imgs, sub_img_size)
raw_data = getRawData(no_imgs);
[dataset, labels] = dataLoop(raw_data, sub_img_size);

