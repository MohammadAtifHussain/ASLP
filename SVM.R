library(corrplot)
library(caret)
library(e1071)
library(MLmetrics)

traindata_raw <- read.csv("train.csv")
traindata <- traindata_raw
traindata$id <- NULL
testdata_raw <- read.csv("test.csv")
testdata <- testdata_raw
testdata$id <- NULL


for (i in 1:116) {
  traindata[,i] <- as.numeric(traindata[,i])
  testdata[,i] <- as.numeric(testdata[,i])
}


#################################################################################
mean_traindata <- lapply(traindata , mean)
sd_traindata <- lapply(traindata , sd)
mean_traindata <- unlist(mean_traindata)
sd_traindata <- unlist(sd_traindata)
train_mean_cntrd = as.data.frame(scale(traindata, center = mean_traindata, scale = sd_traindata))
train_mean_cntrd$loss <- traindata$loss

mean_testdata <- lapply(testdata , mean)
sd_testdata <- lapply(testdata , sd)
mean_testdata <- unlist(mean_testdata)
sd_testdata <- unlist(sd_testdata)
test_mean_cntrd = as.data.frame(scale(testdata, center = mean_testdata, scale = sd_testdata))


correlationMatrix_1 <- cor(train_mean_cntrd[,1:130] , train_mean_cntrd$loss)
correlated_attr <- which(abs(correlationMatrix_1) > 0.01)

train_corr_attr <- train_mean_cntrd[,correlated_attr]

test_corr_attr <- test_mean_cntrd[,correlated_attr]

correlationMatrix_final <- cor(train_corr_attr)
highlyCorrelated <- findCorrelation(correlationMatrix_final, cutoff=0.7)
train_final <- train_corr_attr[,-highlyCorrelated]
train_final$loss <- train_mean_cntrd$loss

test_final <- test_corr_attr[,-highlyCorrelated]

#list of models
model_list_6 <- vector(mode = "list", length = 10)

start_index <- 1
for (i in 1:10) 
{
  print(i)
  if(i<10)
  {
    print(start_index)
    model_list_6[[i]] <- svm(loss~. , data = train_final[start_index : (start_index+20000),], kernel = "polynomial", degree = 2, gamma = 0.05,  coef0 = 0.5, epsilon = 0.2 , cost = 10)
    start_index <- start_index + 20000
  }
  else
  {
    print(start_index+8317)
    model_list_6[[i]] <- svm(loss~. , data = train_final[start_index : (start_index+8318),], kernel = "polynomial", degree = 2, gamma = 0.05,  coef0 = 0.5, epsilon = 0.2 , cost = 10)
  }
}

save(model_list_6, file = "model_list_6.rda")

################## predict on all the instances of training data..... and calculate over all r2 value

result_matrix_train <- matrix(data = 0 , nrow = nrow(train_final) , ncol = 10)
result_avg_train <- matrix(data = 0 , nrow = nrow(train_final) , ncol = 1)

for (i in 1:10) 
{
  print(i)
  result_matrix_train[,i] <- predict(model_list_6[[i]] , train_final)
}
for (i in 1:nrow(train_final)) 
{
  result_avg_train[i] = mean(result_matrix_train[i,])
}
R_sqrd_train <- R2_Score(result_avg_train , train_final$loss)
print(R_sqrd_train)# value obtained ==> 0.5298311


##################predict on all the instances of test data

result_matrix_test <- matrix(data = 0 , nrow = nrow(test_final) , ncol = 10)
result_avg_test <- matrix(data = 0 , nrow = nrow(test_final) , ncol = 1)

for (i in 1:10) 
{
  print(i)
  result_matrix_test[,i] <- predict(model_list_6[[i]] , test_final)
}
for (i in 1:nrow(test_final)) 
{
  result_avg_test[i] = mean(result_matrix_test[i,])
}

