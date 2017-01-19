library(rpart)
library(corrplot)
library(caret)
library(e1071)
library(MLmetrics)

traindata_raw <- read.csv("C://books//Machine learning//data//train.csv")
traindata <- traindata_raw
traindata$id <- NULL


for (i in 1:116) {
  traindata[,i] <- as.numeric(traindata[,i])
}


#################################################################################
mean_traindata <- lapply(traindata , mean)
sd_traindata <- lapply(traindata , sd)
mean_traindata <- unlist(mean_traindata)
sd_traindata <- unlist(sd_traindata)
train_mean_cntrd = as.data.frame(scale(traindata, center = mean_traindata, scale = sd_traindata))
train_mean_cntrd$loss <- traindata$loss

correlationMatrix_1 <- cor(train_mean_cntrd[,1:130] , train_mean_cntrd$loss)
correlated_attr <- which(abs(correlationMatrix_1) > 0.01)
#plot(correlationMatrix_1)
train_corr_attr <- train_mean_cntrd[,correlated_attr]

correlationMatrix_final <- cor(train_corr_attr)
highlyCorrelated <- findCorrelation(correlationMatrix_final, cutoff=0.7)
train_final <- train_corr_attr[,-highlyCorrelated]
train_final$loss <- train_mean_cntrd$loss

#list of models
model_listR<- vector(mode = "list", length = 10)
result_matrix <- matrix(data = 0 , nrow = 20001 , ncol = 10)

start_index <- 1
for (i in 1:10) 
{
  
  if(i<10)
  {
   
    model_listR[[i]]<-randomForest(loss~.,data = train_final[start_index : (start_index+20000),],mtry=15,ntree=700,nodesize=4,na.action=na.omit)
    
    print(start_index)
    start_index <- start_index + 20000
  }
  else
  {
   
    model_listR[[i]]<-randomForest(loss~.,data = train_final[start_index : (start_index+8317),],mtry=15,ntree=900,nodesize=6,na.action=na.omit)
    print(start_index+8317)
   
  }

  
}
prediction<-predict(model_listR, train_final[1:10000,])

r2 <- R2_Score(prediction , train_final[1:10000,ncol(train_final)])

start_index <- 1
result_matrix <- matrix(data = 0 , nrow = 20001 , ncol = 10)
R_sqrd_val <- matrix(data = 0 , nrow = 10 ,ncol = 1)
for (i in 1:10) 
{
  print(i)
  if(i<10)
  {
   
    result_matrix[,i] <- predict(model_listR[[i]] , train_final[start_index :(start_index+20000),])
    R_sqrd_val[i] <- R2_Score(result_matrix[,i] , train_final[start_index :(start_index+20000),ncol(train_final)])
    start_index <- start_index + 20000
  }
  else
  {
    print(start_index+8317)
    
    result_temp <- predict(model_listR[[i]] , train_final[start_index :(start_index+8317),])
    R_sqrd_val[i] <- R2_Score(result_temp , train_final[start_index :(start_index+8317),ncol(train_final)])
  }
}

print(R_sqrd_val)

Average_R_squared <- mean(R_sqrd_val)
print(Average_R_squared)
