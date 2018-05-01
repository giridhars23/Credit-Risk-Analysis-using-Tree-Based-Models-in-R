  set.seed(123)

library(rpart)
library(Metrics)
library(randomForest)
library(caret)
library(ipred)

dataset <- read.csv("messy_stock_data.csv")

n <- nrow(dataset)

n_train <- round(0.8 * n) 

train_indices <- sample(1:n, n_train)

credit_train <- dataset[train_indices, ]  

credit_test <- dataset[-train_indices, ]  

######Decision Tree######

credit_model <- rpart(formula = default ~ ., 
                      data = credit_train, 
                      method = "class")

pred <- predict(object = best_model,
                newdata = credit_test,type="prob")

auc(actual = ifelse(credit_test$default == "yes", 1, 0), 
    predicted = pred[,"yes"]) 

# Establish a list of possible values for minsplit and maxdepth
minsplit <- seq(2, 8, 1)
maxdepth <- seq(1, 20, 1)

# Create a data frame containing all combinations 
hyper_grid <- expand.grid(minsplit = minsplit, maxdepth = maxdepth)


# Number of potential models in the grid
num_models <- nrow(hyper_grid)

# Create an empty list to store models
credit_models <- list()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:num_models) {
  
  # Get minsplit, maxdepth values at row i
  minsplit <- hyper_grid$minsplit[i]
  maxdepth <- hyper_grid$maxdepth[i]
  
  # Train a model and store in the list
  credit_models[[i]] <- rpart(formula = default ~ ., 
                              data = credit_train, 
                              method = "class",
                              minsplit = minsplit,
                              maxdepth = maxdepth)
}

num_models <- length(credit_models)

# Create an empty vector to store RMSE values
roc <- c()

# Write a loop over the models to compute validation RMSE
for (i in 1:num_models) {
  
  # Retreive the i^th model from the list
  model <- credit_models[[i]]
  
  # Generate predictions on grade_valid 
  pred <- predict(object = credit_models[[i]],
                  newdata = credit_test)
  
  # Compute validation RMSE and add to the 
  roc[i] <- auc(actual = ifelse(credit_test$default == "yes", 1, 0), 
                predicted = pred[,"yes"])       
}

# Identify the model with smallest validation set RMSE
best_model <- credit_models[[which.max(roc)]]

pred <- predict(object = best_model,
                newdata = credit_test,type="prob")

auc(actual = ifelse(credit_test$default == "yes", 1, 0), 
    predicted = pred[,"yes"]) 


plotcp(best_model)

print(best_model$cptable)

opt_index <- which.min(best_model$cptable[, "xerror"])

cp_opt <- best_model$cptable[opt_index, "CP"]

credit_model_prunned <- prune(tree = best_model, 
                              cp = cp_opt)


pred <- predict(object = credit_model_prunned,
                newdata = credit_test,type="class")

confusionMatrix(data = pred,       
                reference = credit_test$default)  

pred <- predict(object = credit_model_prunned,
                newdata = credit_test,type="prob")


dt_auc <- auc(actual = ifelse(credit_test$default == "yes", 1, 0), 
    predicted = pred[,"yes"]) 


##########Random Forest###########


random_model <- randomForest(formula = default ~ ., 
                             data = credit_train)


err <- random_model$err.rate


# Look at final OOB error rate (last row in err matrix)
oob_err <- err[nrow(err), "OOB"]

pred <- predict(object = random_model,
                newdata = credit_test,type="class")

cm <- confusionMatrix(data = pred,       
                reference = credit_test$default )  


# Establish a list of possible values for mtry, nodesize and sampsize
mtry <- seq(4, ncol(credit_train) * 0.8, 2)
nodesize <- seq(3, 8, 2)
sampsize <- nrow(credit_train) * c(0.7, 0.8)

# Create a data frame containing all combinations 
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, sampsize = sampsize)

# Create an empty vector to store OOB error values
oob_err <- c()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid)) {
  
  # Train a Random Forest model
  model <- randomForest(formula = default ~ ., 
                        data = credit_train,
                        mtry = hyper_grid$mtry[i],
                        nodesize = hyper_grid$nodesize[i],
                        sampsize = hyper_grid$sampsize[i])
  
  # Store OOB error for the model                      
  oob_err[i] <- model$err.rate[nrow(model$err.rate), "OOB"]
}

# Identify optimal set of hyperparmeters based on OOB error
opt_i <- which.min(oob_err)
print(hyper_grid[opt_i,])

rf_model <- randomForest(formula = default ~ ., 
                      data = credit_train,
                      mtry = 6,
                      nodesize = 5,
                      sampsize = 560)

pred <- predict(object = rf_model,
                newdata = credit_test,type="class")

confusionMatrix(data = pred,       
                reference = credit_test$default)  

pred <- predict(object = rf_model,
                newdata = credit_test,type="prob")


rf_auc <- auc(actual = ifelse(credit_test$default == "yes", 1, 0), 
              predicted = pred[,"yes"]) 

#######Bagging#######

credit_model <- bagging(formula = default ~ ., 
                        data = credit_train,
                        coob = TRUE)

class_prediction <- predict(object = credit_model,    
                            newdata = credit_test,  
                            type = "class")  # return classification labels


confusionMatrix(data = class_prediction,       
                reference = credit_test$default)  

# Generate predictions on the test set
pred <- predict(object = credit_model,
                newdata = credit_test,
                type = "prob")

# Compute the AUC 
auc(actual = ifelse(credit_test$default == "yes", 1, 0), 
    predicted = pred[,"yes"])                    

ctrl <- trainControl(method = "cv",     # Cross-validation
                     number = 5,      
                     classProbs = TRUE,                  
                     summaryFunction = twoClassSummary)  

# Cross validate the credit model using "treebag" method; 
# Track AUC (Area under the ROC curve)
credit_caret_model <- train(default ~ .,
                            data = credit_train, 
                            method = "treebag",
                            metric = "ROC",
                            trControl = ctrl)


print(credit_caret_model)


names(credit_caret_model)

# Print the CV AUC
credit_caret_model$results[,"ROC"]

pred <- predict(object = credit_caret_model, 
                newdata = credit_test,
                type = "raw")


confusionMatrix(data = pred,       
                reference = credit_test$default)  

pred <- predict(object = credit_caret_model, 
                newdata = credit_test,
                type = "prob")

# Compute the AUC (`actual` must be a binary (or 1/0 numeric) vector)
bag_auc <- auc(actual = ifelse(credit_test$default == "yes", 1, 0), 
    predicted = pred[,"yes"])

############Gradient Boosted Trees - GBM##################


credit_train$default <- ifelse(credit_train$default == "yes", 1, 0)

credit_model <- gbm(formula = default ~ ., 
                    distribution = "bernoulli", 
                    data = credit_train,
                    n.trees = 10000)

print(credit_model)
summary(credit_model)

preds <- predict(object = credit_model, 
                  newdata = credit_test,
                  n.trees = 10000,
                  type = "response")

credit_test$default <- ifelse(credit_test$default == "yes", 1, 0)

auc(actual = credit_test$default, predicted = preds ) 

ntree_opt_oob <- gbm.perf(object = credit_model, 
                          method = "OOB", 
                          oobag.curve = TRUE)

credit_model_cv <- gbm(formula = default ~ ., 
                       distribution = "bernoulli", 
                       data = credit_train,
                       n.trees = 10000,
                       cv.folds = 2)

# Optimal ntree estimate based on CV
ntree_opt_cv <- gbm.perf(object = credit_model_cv, 
                         method = "cv")

# Generate predictions on the test set using ntree_opt_oob number of trees
preds1 <- predict(object = credit_model, 
                  newdata = credit_test,
                  n.trees = ntree_opt_oob)

# Generate predictions on the test set using ntree_opt_cv number of trees
preds2 <- predict(object = credit_model, 
                  newdata = credit_test,
                  n.trees = ntree_opt_cv)   

# Generate the test set AUCs using the two sets of preditions & compare
auc1 <- auc(actual = credit_test$default, predicted = preds1)  #OOB
auc2 <- auc(actual = credit_test$default, predicted = preds2)

gbm_auc <- max(auc1,auc2)



#############Results###############

sprintf("Decision Tree Test AUC: %.3f", dt_auc)
sprintf("Bagged Trees Test AUC: %.3f", bag_auc)
sprintf("Random Forest Test AUC: %.3f", rf_auc)
sprintf("GBM Test AUC: %.3f", gbm_auc)

