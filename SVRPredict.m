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
%-s svm���ͣ�SVM��������(Ĭ��0) 0 -- C-SVC 1 --v-SVC 2 -- һ��SVM  3 -- e -SVR  4 -- v-SVR
%-t��ʾѡ��ĺ˺������ͣ�-t=0ʱ���Ժˣ�-t=1����ʽ�ˣ�-t=2���������������˹����-t=3��sigmod�˺�����-t=4��Ԥ�����
%-cΪ�ͷ�����ϵ��
%-gΪ�˺����е�gamma��������(��Զ���ʽ/rbf/sigmoid�˺���)(Ĭ��1/k)
%-p����e -SVR ����ʧ����p��ֵ(Ĭ��0.1)

test_x_data = all_data((train_num + 1) : all_data_num, 1 : data_dimension);
test_y_data = all_data((train_num + 1) : all_data_num, (data_dimension + 1));
% SVR Predict
[predicted_label2, accuracy2, prob_estimates2] = svmpredict(test_y_data, test_x_data, model);
predicted_y = predicted_label2;


