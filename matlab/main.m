%% Add path with matconvnet library functions 
addpath('/home/matilda/Documents/matconvnet/matconvnet-1.0-beta25/matlab');
addpath(genpath('../crchistophenotypes_2016_04_28/CRCHistoPhenotypes_2016_04_28/Classification'));
%% Define data, no extra preprocessing so far
no_imgs = 100; sub_img_size = 27;
[images, labels, raw_data] = getData(no_imgs, sub_img_size);

%% test matconvnet
% if it does not work, try: vl_compilenn('EnableImreadJpeg', false)
% bank of linear filters
images = im2single(images);
x = images(:,:,:,1);
%% C layer 1
w = randn(4,4,1,36,'single');
y = vl_nnconv(x,w,[]);
z = vl_nnrelu(y);
%% M layer 1
% stride is based on   YH = floor((H + (PADTOP+PADBOTTOM) - POOLY)/STRIDEY)
% + 1, which is to get the correct dimensions out, according to report
x = vl_nnpool(z, 2, 'Stride', 2);

%% C layer 2
w = randn(3,3,36,48,'single');
y = vl_nnconv(x,w,[]);
z = vl_nnrelu(y);


%% M layer 2
x = vl_nnpool(z, 2, 'Stride', 2);

%% F layer 1 (OBS! No dropout)
w = randn(5,5,48,512,'single');
y = vl_nnconv(x,w,[]);
z = vl_nnrelu(y);

%% F layer 2 (OBS! No dropout)

w = randn(1,1,512,512,'single');
y = vl_nnconv(z,w,[]);
z = vl_nnrelu(y);

%% F layer 3 (final layer)

w = randn(1,1,512,4,'single');
y = vl_nnconv(z,w,[]);
z = vl_nnrelu(y);
