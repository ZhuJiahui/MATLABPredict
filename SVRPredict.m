clear all;
clc;

% Totally 28049
all_data = load('origin_data.txt');
all_data_num = size(all_data, 1);
data_dimension = size(all_data, 2) - 1;

train_num = round(all_data_num * 0.8);
train_x_data = all_data(1 : train_num, 1 : data_dimension);
train_y_data = all_data(1 : train_num, (data_dimension + 1));

% SVR Model
model = svmtrain(train_y_data, train_x_data, '-s 3 -t 2 -c 0.5 -g 2.5 -p 0.005');
%-s svm类型：SVM设置类型(默认0) 0 -- C-SVC 1 --v-SVC 2 -- 一类SVM  3 -- e -SVR  4 -- v-SVR
%-t表示选择的核函数类型，-t=0时线性核；-t=1多项式核；-t=2，径向基函数（高斯）；-t=3，sigmod核函数；-t=4，预计算核
%-c为惩罚因子系数
%-g为核函数中的gamma函数设置(针对多项式/rbf/sigmoid核函数)(默认1/k)
%-p设置e -SVR 中损失函数p的值(默认0.1)

test_x_data = all_data((train_num + 1) : all_data_num, 1 : data_dimension);
test_y_data = all_data((train_num + 1) : all_data_num, (data_dimension + 1));
% SVR Predict
[predicted_label2, accuracy2, prob_estimates2] = svmpredict(test_y_data, test_x_data, model);
predicted_y = predicted_label2;


