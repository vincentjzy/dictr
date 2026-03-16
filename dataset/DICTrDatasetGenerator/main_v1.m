clc, clear, close all;

%% general setting
source_dir = './StrainNetSample';
files = dir(fullfile(source_dir, '*.tif'));
train_dir = '../train';
eval_dir = '../eval';
if ~exist(train_dir, 'dir')
    mkdir(train_dir);
end
if ~exist(eval_dir, 'dir')
    mkdir(eval_dir);
end

% output img size
width = 128;
height = 128;

% outer range for interpolation
border = 8;
vWidth = width + border * 2;
vHeight = height + border * 2;

% will execute twice get 3200 train and 320 eval
train_size = 1600;
eval_size = 160;

% noise parameters measured from the Blackfly camera BFS-U3-31S4M-C
myparamnoise=paramnoise('S',0.02019,0.39221);

% first sample, small displacement
grid_sizes1 = [4,8,16,32,64];
dMiu1 = 0;
dSigma1 = 0.5;

% second sample, large displacement
grid_sizes2 = [24, 36, 48, 60];
dMiu2 = 0;
dSigma2 = 2.5;

%% start parallel
c = parcluster('local');
c.NumWorkers = maxNumCompThreads;
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool(c, c.NumWorkers);
end
disp(['Start ',num2str(c.NumWorkers),' parallel works']);

%% generation
% first train dataset
parfor i = 1 : train_size
    j = mod(i, 363) + 1;
    k = mod(i, 5) + 1;
    src_name = sprintf('%s/%s', source_dir, files(j).name);
    fullimg = imread(src_name);

    s=rng('shuffle');
    x_start = randi([1, size(fullimg,2) - vWidth + 1]);
    y_start = randi([1, size(fullimg,1) - vHeight + 1]);
    
    ref=fullimg(y_start:y_start + vHeight - 1, x_start:x_start + vWidth - 1, :);
    ref_noise = addnoise(myparamnoise, 8, ref);

    deform = generate_deform(vWidth, vHeight, grid_sizes1(k), dMiu1, dSigma1);
    tar = warp_img(ref, deform);
    tar_noise = addnoise(myparamnoise, 8, tar);

    ref_name_noise = sprintf('%s/REF%05d.bmp', train_dir, i-1);
    imwrite(uint8(ref_noise(border:border+height-1, border:border+width-1)), ref_name_noise);

    tar_name_noise = sprintf('%s/TAR%05d.bmp', train_dir, i-1);
    imwrite(uint8(tar_noise(border:border+height-1, border:border+width-1)), tar_name_noise);

    def_name = sprintf('%s/DEF%05d.bin', train_dir, i-1);
    fid = fopen(def_name, 'wb');
    fwrite(fid, permute(deform(border:border+height-1, border:border+width-1, :), [2, 1, 3]), 'float');
    fclose(fid); 
end

% first eval dataset
parfor i = 1 : eval_size
    j = mod(i, 363) + 1;
    k = mod(i, 5) + 1;
    src_name = sprintf('%s/%s', source_dir, files(j).name);
    fullimg = imread(src_name);

    s=rng('shuffle');
    x_start = randi([1, size(fullimg,2) - vWidth + 1]);
    y_start = randi([1, size(fullimg,1) - vHeight + 1]);
    
    ref=fullimg(y_start:y_start + vHeight - 1, x_start:x_start + vWidth - 1, :);
    ref_noise = addnoise(myparamnoise, 8, ref);

    deform = generate_deform(vWidth, vHeight, grid_sizes1(k), dMiu1, dSigma1);
    tar = warp_img(ref, deform);
    tar_noise = addnoise(myparamnoise, 8, tar);

    ref_name_noise = sprintf('%s/REF%05d.bmp', eval_dir, i-1);
    imwrite(uint8(ref_noise(border:border+height-1, border:border+width-1)), ref_name_noise);

    tar_name_noise = sprintf('%s/TAR%05d.bmp', eval_dir, i-1);
    imwrite(uint8(tar_noise(border:border+height-1, border:border+width-1)), tar_name_noise);

    def_name = sprintf('%s/DEF%05d.bin', eval_dir, i-1);
    fid = fopen(def_name, 'wb');
    fwrite(fid, permute(deform(border:border+height-1, border:border+width-1, :), [2, 1, 3]), 'float');
    fclose(fid); 
end

% second train dataset
parfor i = 1 : train_size
    j = mod(i, 363) + 1;
    k = mod(i, 4) + 1;
    src_name = sprintf('%s/%s', source_dir, files(j).name);
    fullimg = imread(src_name);

    s=rng('shuffle');
    x_start = randi([1, size(fullimg,2) - vWidth + 1]);
    y_start = randi([1, size(fullimg,1) - vHeight + 1]);
    
    ref=fullimg(y_start:y_start + vHeight - 1, x_start:x_start + vWidth - 1, :);
    ref_noise = addnoise(myparamnoise, 8, ref);

    deform = generate_deform(vWidth, vHeight, grid_sizes2(k), dMiu2, dSigma2);
    tar = warp_img(ref, deform);
    tar_noise = addnoise(myparamnoise, 8, tar);

    ref_name_noise = sprintf('%s/REF%05d.bmp', train_dir, i-1+train_size);
    imwrite(uint8(ref_noise(border:border+height-1, border:border+width-1)), ref_name_noise);

    tar_name_noise = sprintf('%s/TAR%05d.bmp', train_dir, i-1+train_size);
    imwrite(uint8(tar_noise(border:border+height-1, border:border+width-1)), tar_name_noise);

    def_name = sprintf('%s/DEF%05d.bin', train_dir, i-1+train_size);
    fid = fopen(def_name, 'wb');
    fwrite(fid, permute(deform(border:border+height-1, border:border+width-1, :), [2, 1, 3]), 'float');
    fclose(fid); 
end

% second eval dataset
parfor i = 1 : eval_size
    j = mod(i, 363) + 1;
    k = mod(i, 4) + 1;
    src_name = sprintf('%s/%s', source_dir, files(j).name);
    fullimg = imread(src_name);

    s=rng('shuffle');
    x_start = randi([1, size(fullimg,2) - vWidth + 1]);
    y_start = randi([1, size(fullimg,1) - vHeight + 1]);
    
    ref=fullimg(y_start:y_start + vHeight - 1, x_start:x_start + vWidth - 1, :);
    ref_noise = addnoise(myparamnoise, 8, ref);

    deform = generate_deform(vWidth, vHeight, grid_sizes2(k), dMiu2, dSigma2);
    tar = warp_img(ref, deform);
    tar_noise = addnoise(myparamnoise, 8, tar);

    ref_name_noise = sprintf('%s/REF%05d.bmp', eval_dir, i-1+eval_size);
    imwrite(uint8(ref_noise(border:border+height-1, border:border+width-1)), ref_name_noise);

    tar_name_noise = sprintf('%s/TAR%05d.bmp', eval_dir, i-1+eval_size);
    imwrite(uint8(tar_noise(border:border+height-1, border:border+width-1)), tar_name_noise);

    def_name = sprintf('%s/DEF%05d.bin', eval_dir, i-1+eval_size);
    fid = fopen(def_name, 'wb');
    fwrite(fid, permute(deform(border:border+height-1, border:border+width-1, :), [2, 1, 3]), 'float');
    fclose(fid); 
end

%% quit parallel
delete(gcp('nocreate'));