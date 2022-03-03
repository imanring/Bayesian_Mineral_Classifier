load('C:\Users\isaac\Documents\Quiz and Homework\CSCI\Machine Learning\CRISM_labeled_pixels_ratioed.mat');
c = size(unique(pixlabs),1);
%fix labels
pixlabs(pixlabs==39) = 22;
%split data into test and train sets
train_spec = pixspec(pixims < 56, 1:248);
train_labs = pixlabs(pixims < 56);
test_spec = pixspec(pixims > 55, 1:248);
test_labs = pixlabs(pixims > 55);
%number of samples
n = size(train_spec,1);
%dimension
d = size(train_spec,2);

%split into categories
X = cell(c);
for i = 1:c
    X{i} = train_spec(train_labs==i,:);
end

x_ = zeros(c,d);
S = zeros(c,d,d);
%train
for c_i = 1:c
    x_(c_i,:) = mean(X{c_i});
    n = size(X{c_i},1);
    S(c_i,:,:) = (n-1)/n*cov(X{c_i});
end


%find hyperparameters
mu_0 = mean(x_);
m = d^3;
sigma_0_temp = reshape(mean(S),[d,d]);
sigma_0 = sigma_0_temp*(m-d-1);
k = 30;

%predict
likelihood = zeros(c,size(test_spec,1));
x = test_spec;
for c_i = 1:c
    mean_t = (n*x_(c_i,:)+k*mu_0)/(n+k);
    shape = (n+k+1)/((n+k)*(n+m+1-d))*(sigma_0+(n-1)*reshape(S(1,:,:),d,d)+n*k/(n+k)*(mu_0-x_(c_i,:))'*(mu_0-x_(c_i,:)));
    likelihood(c_i,:) = mvtpdf(x-mean_t,shape,round(m+n));
end
[l, max_ind] = max(likelihood);
accuracy = sum(max_ind'==test_labs)/size(test_labs,1);
disp(accuracy)

class_ac = 0;
leng = 0;
for i = 1:c
    if sum(test_labs==i) ~= 0
        class_ac = class_ac + (max_ind==i)*(test_labs==i)/sum(test_labs==i);
        leng = leng+1;
    end
end
disp(class_ac/leng)
