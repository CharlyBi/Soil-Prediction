%matlab script for "Africa Soil Property Prediction Challenge" competition
%
% Author:
%   Charly Bitton, 2014
%
% License:
%   This software is distributed under the GNU General Public License 
%   (version 2 or later); please refer to the file LICENSE.txt, included 
%   with the software, for details. 

clear all

rng('default');
data_path = '..\data\';

%reading training data
x = dataset('file',[data_path 'training.csv'],'Delimiter',',');
x.Depth = strcmp(x.Depth,'Topsoil');
y = single(x(:,end-4:end));
x = single(x(:,2:end-5));

%reading test data
x_tst = dataset('file',[data_path 'sorted_test.csv'],'Delimiter',',');
x_tst.Depth = strcmp(x_tst.Depth,'Topsoil');
tst_id = x_tst.PIDN;
x_tst = single(x_tst(:,2:end));

% train feature dilution
w = 0.54 - 0.46*cos(2*pi*(0:7)/15);
ham_w   = [w w(end:-1:1)];           % 16-tap hamming windows
ham_w   = ham_w / sum(ham_w);
x_train = smooth0(ham_w, x', 8)';    % 0 delay smoothing
x_train = x_train(:, 1:8:end);
x_train = x_train(:, [1:40 100:end]);

% test feature dilution
x_test = smooth0(ham_w, x_tst', 8)'; %0 delay smoothing
x_test = x_test(:, 1:8:end);
x_test = x_test(:, [1:40 100:end]);

%normilezd train and test set #1
x_train_1  = (x_train - 0.8)/0.2;
x_test_1   = (x_test - 0.8)/0.2;

%normilezd train and test set #2
x_train_2  = (diff(x_train')')/0.015;
x_test_2   = (diff(x_test')')/0.015;


% create weight vector w (used for set #2 only)
w_std = std(x_train_2);
w_std = (w_std - min(w_std))/max(w_std);

% apply weight to data
x_train_2 = bsxfun(@times,x_train_2,w_std);
x_test_2  = bsxfun(@times,x_test_2,w_std);



% model building
% 
% x_train_1 / x_test_1 used for target #5 prediction
% 
% x_train_2 / x_test_2 used for target #1-4 prediction
%
%


x_train  = x_train_2;

x_test   = x_test_2;

v = zeros(size(x_test,1),5); %allocate output result


for k = 1:5 %loop over 5 targets

        if k==5, % target 5, Sand

                x_train = x_train_1;
                
                x_test  = x_test_1;
                
        end
        
        cv = cvpartition(y(:,1)*0,'k',5);

        %array for all 20 models results for target k
        y_target_all = zeros(size(x_test,1),20); 

        for m = 1:20                      %loop averaging 20 CV  

                y_target = x_test(:,1)*0; %init test results with 0

                for n = 1:cv.NumTestSets  % averaging 5 CV models

                        ii = training(cv,n);

                        %define the network 1-4-4-1
                        net = fitnet([4 4]);

                        %limit the number of max epoch
                        net.trainParam.epochs = 100;

                        %train NN with levenberg-marquet algorithm
                        net = trainlm(net, x_train(ii,:)',y(ii,k)');

                        %apply the network to the test data and acc.
                        y_target = y_target + net(x_test')';
		end

                %average all 5 CV models
                y_target_all(:,m) = y_target/cv.NumTestSets;
                
        end

        disp(['Done target ' num2str(k)])

        v(:,k) = mean(y_target_all')'; %average results
end


%submission
fd = fopen('My_Submission.csv','w')

fprintf(fd,'PIDN,Ca,P,pH,SOC,Sand\n');

for k = 1:size(v,1)

        fprintf(fd,'%s,%1.4f,%1.4f,%1.4f,%1.4f,%1.4f\n',tst_id{k},v(k,:));

end

fclose(fd);

