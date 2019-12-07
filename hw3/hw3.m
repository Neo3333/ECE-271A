load('TrainingSamplesDCT_8_new.mat');
train_FG = TrainsampleDCT_FG;
train_BG = TrainsampleDCT_BG;

a = randi(1,8);
a = a / sum(a);