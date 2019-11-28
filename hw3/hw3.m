train_set = load('hw3Data/TrainingSamplesDCT_subsets_8.mat');
alpha = load('hw3Data/Alpha.mat');
alpha = alpha.alpha;
strategy = load('hw3Data/Prior_1.mat');
d1_BG = train_set.D1_BG;
d1_FG = train_set.D1_FG;
bayes_error = [];
mle_error = [];
map_error = [];
n_FG = size(d1_FG,1);
n_BG = size(d1_BG,1);

% Loop for different alpha
for alpha_idx = 1:2
    cov_0 = zeros(64,64);
    for idx = 1:64
       cov_0(idx,idx) = alpha(alpha_idx)*strategy.W0(idx); 
    end

    % FG
    d1_FG_cov = cov(d1_FG) * (n_FG-1)/n_FG;
    tmp2 = inv(cov_0 + (1/n_FG)*d1_FG_cov);
    mu_1_FG = cov_0 * tmp2 * transpose(mean(d1_FG)) + (1/n_FG) * d1_FG_cov * tmp2 * transpose(strategy.mu0_FG);
    cov_1_FG = cov_0 * tmp2 * (1/n_FG) * d1_FG_cov;
    % predictive distribution (normal distribution)
    mu_pred_FG = mu_1_FG;
    cov_pred_FG = d1_FG_cov + cov_1_FG;

    % BG
    d1_BG_cov = cov(d1_BG) * (n_BG-1)/n_BG;
    tmp3 = inv(cov_0 + (1/n_BG)*d1_BG_cov);
    mu_1_BG = cov_0 * tmp3 * transpose(mean(d1_BG)) + (1/n_BG) * d1_BG_cov * tmp3 * transpose(strategy.mu0_BG);
    cov_1_BG = cov_0 * tmp3 * (1/n_BG) * d1_BG_cov;
    % predictive distribution (normal distribution)
    mu_pred_BG = mu_1_BG;
    cov_pred_BG = d1_BG_cov + cov_1_BG;

    % Prior
    num_FG = size(d1_FG,1);
    num_BG = size(d1_BG,1);
    prior_FG = num_FG / (num_FG + num_BG);
end
img = imread('../homework1/cheetah.bmp');
img = im2double(img);
save('im_double.mat','img');
d = [2,3,4;5,6,7;8,9,10]
e = dct2(d)
DCT = dct2(img(1:1+7,1:1+7));
zigzag_order = zigzag(DCT);