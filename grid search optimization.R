# Import all the neccessary libraries
library(caret)
library(pROC)
library(e1071)  
library(randomForest)
library(kernlab)
library(gbm)
data<-read.csv("C:\\Users\\Janhavi\\OneDrive\\Desktop\\Coimbra_breast_cancer_dataset.csv")# import the data 
str(data)# structure data
# converting into factors and levels
Classification <- colnames(data)[ncol(data)]
data[, Classification] <- as.factor(data[[Classification]])
data[[Classification]] <- factor(data[[Classification]], levels = c("0", "1"), labels = c("Class0", "Class1"))
data[[Classification]] <- make.names(as.factor(data[[Classification]]))
levels(data[[Classification]])
set.seed(42) # setting the seed for reproducability
trainIndex <- createDataPartition(data[[Classification]], p = 0.8, list = FALSE)#partitioning the data 80%training and 20%test set
traindata <- data[trainIndex, ]#train set
testdata <- data[-trainIndex, ]#test set
#5 Fold Cross-validation
trainControl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
traindata$Classification <- as.factor(traindata$Classification)
testdata$Classification<-as.factor(testdata$Classification)
levels(testdata$Classification) <- levels(traindata$Classification)
#logistic regression model
logisticGrid <- expand.grid(alpha = c(0, 0.5, 1),lambda = seq(0, 0.1, length = 10))
logisticModel <- train(Classification ~ ., data = traindata, method = "glmnet", tuneGrid = logisticGrid, trControl = trainControl)
# for AUC of logistic regression model
log_model <- train(
  formula(paste(Classification, "~ .")),
  data = traindata,
  method = "glm",
  family = "binomial",
  trControl = trainControl,
  metric = "ROC"
)
# evaluate the model and predict the accuracy of logistic regression
logistic_predictions <- predict(logisticModel, testdata)
conf_lm<-confusionMatrix(logistic_predictions,testdata$Classification)
accuracy_lm <- conf_lm$overall['Accuracy']
accuracy_lm
# Random Forest Classifier
rf_grid <- expand.grid(mtry = c(2, 4, 6))
rf_model<-train(Classification~.,data=traindata,method="rf",tuneGrid=rf_grid,trControl=trainControl)
# for AUC 
random_model <- train(
  formula(paste(Classification, "~ .")),
  data = traindata,
  method = "rf",
  tuneGrid = rf_grid,
  trControl = trainControl,
  metric = "ROC"
)
#evaluate the model and predict the accuracy of random forest classifier
rf_predictions <- predict(rf_model, testdata)
rf_predictions <- factor(rf_predictions, levels = levels(testdata$Classification))
conf_rf<-confusionMatrix(rf_predictions,testdata$Classification)
accuracy_rf <- conf_rf$overall['Accuracy']
accuracy_rf
#decision tree model
dt_grid <- expand.grid(cp = c(0.01, 0.05, 0.1))
dt_model <- train(Classification ~ ., data = traindata, method="rpart",tuneGrid=dt_grid,trControl=trainControl)
# for AUC
tree_model <- train(
  formula(paste(Classification, "~ .")),
  data = traindata,
  method = "rpart",
  tuneLength = 10,
  trControl = trainControl,
  metric = "ROC"
)
#evaluate the model and predict the acccuracy of decision tree
dt_predictions<-predict(dt_model,testdata)
conf_dt<-confusionMatrix(dt_predictions,testdata$Classification)
accuracy_dt<-conf_dt$overall['Accuracy']
accuracy_dt
#Support Vector Machine
svm_grid <- expand.grid(C = c(0.1, 1, 10), sigma = c(0.01, 0.05, 0.1))
svm_model <- train(Classification ~ ., data = traindata, method="svmRadial",tuneGrid=svm_grid,trControl=trainControl)
# for AUC
supp_model <- train(
  formula(paste(Classification, "~ .")),
  data = traindata,
  method = "svmRadial",
  tuneGrid = svm_grid,
  trControl = trainControl,
  metric = "ROC"
)
#evaluate the model and predict the accuracy of SVM
svm_predictions<-predict(svm_model,testdata)
conf_svm<-confusionMatrix(svm_predictions,testdata$Classification)
accuracy_svm<-conf_svm$overall['Accuracy']
accuracy_svm
#Gradient Boosting 
gbm_grid<-expand.grid(interaction.depth=c(1,3,5),n.trees=c(50,100,150),shrinkage=c(0.01,0.1,0.3),n.minobsinnode=c(10,20))
gbm_model <- train(Classification ~ ., data = traindata, method="gbm",tuneGrid=gbm_grid,trControl=trainControl,verbose=FALSE)
# for AUC
gradient_model <- train(
  formula(paste(Classification, "~ .")),
  data = traindata,
  method = "gbm",
  tuneGrid = gbm_grid,
  trControl = trainControl,
  metric = "ROC",
  verbose = FALSE
)
#evaluate the model and predict the accuracy of gradient boosting model
gbm_predictions <- predict(gbm_model, testdata)
gbm_predictions <- factor(gbm_predictions, levels = levels(testdata$Classification))
conf_gbm <- confusionMatrix(gbm_predictions, testdata$Classification)
accuracygrad <- conf_gbm$overall['Accuracy']
accuracygrad
#data frame for Accuracies 
model_accuracy=data.frame(model=c("Logistic Regression", "Random Forest", "Decision Tree", "SVM", "Gradient Boosting"),accuracy=c(accuracy_lm,accuracy_rf,accuracy_dt,accuracy_svm,accuracygrad))
model_accuracy
#  Evaluation of AUC for all models  
evaluate_model <- function(model, testdata, Classification) {
  predictions <- predict(model, newdata = testdata, type = "prob")
  roc_auc <- roc(testdata[[Classification]], predictions[, 2])
  auc(roc_auc)
}

results <- list(
  Logistic_Regression = evaluate_model(log_model, testdata, Classification),
  Decision_Tree = evaluate_model(tree_model, testdata, Classification),
  Random_Forest = evaluate_model(random_model, testdata, Classification),
  SVM = evaluate_model(supp_model, testdata, Classification),
  Gradient_Boosting = evaluate_model(gradient_model, testdata, Classification)
)
results
# Important variable 
varImp(gbm_model)#for Gradient boosting 
varImp(rf_model)# for Random Forest model
varImp(svm_model)#for support vector machine 
varImp(dt_model)#for decision tree model
varImp(logisticModel)#for logistic regression

