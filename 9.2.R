
#------------------------Assignment 9.2 -----------------------------

# Decision Tree - Weight Lifting Excercise

data_set <- read.csv("E:/Data Analytics with RET/Assignment/Example_WearableComputing_weight_lifting_exercises_biceps_curl_variations.csv")
View(data_set)

# remove irrelevant collumns viz. name, cvtd_timestamp, new_window
data <- data_set[,-c(1,4,5)]
View(data)
str(data)

sum(is.na(data))  # there are no missing values

# spliting the data set for train and test

library(caTools)
set.seed(123)
split = sample.split(data$classe, SplitRatio = 0.7) 

train = subset(data, split == TRUE)            # train data
test = subset(data, split == FALSE)            # test data

# a. Create classification model using different decision trees.

library(tree); library(rpart); library(caret); library(C50); library(randomForest)

# Decision Tree
model_tree <- tree(classe ~., data = train)
summary(model_tree)
plot(model_tree); text(model_tree)
pred_tree <- predict(model_tree, test, type = 'class')       # make prediction
conf_tree <- confusionMatrix(test$classe, pred_tree)    # confusion matrix
conf_tree

# CART 
model_cart <- rpart(classe ~ ., data = train)
summary(model_cart)
rpart.plot::rpart.plot(model_cart)
plotcp(model_cart)
pred_cart <- predict(model_cart, test, type = 'class')       # make prediction
conf_cart <- confusionMatrix(test$classe, pred_cart)    # confusion matrix
conf_cart

# CV
train_control <- trainControl(method = "cv", number = 10)
model_cv <- train(classe ~ ., data = train, trControl = train_control, method = "rpart")
model_cv
pred_cv <- predict(model_cv, test)                       # make prediction
conf_cv <- confusionMatrix(test$classe, pred_cv)    # confusion matrix
conf_cv

# Ross Quinlan C5.0
train_control <- trainControl(method = "cv", number = 10)
model_c5.0 <- train(classe ~ ., data = train, trControl = train_control, method = "C5.0")
model_c5.0
pred_c5.0 <- predict(model_c5.0, test)                       # make prediction
conf_c5.0 <- confusionMatrix(test$classe, pred_c5.0)    # confusion matrix
conf_c5.0

# Boosted Tree
train_control <- trainControl(method = "cv", number = 10)
model_bst <- train(classe ~ ., data = train, trControl = train_control, method = "bstTree")
model_bst
pred_bst <- predict(model_bst, test)                       # make prediction
conf_bst <- confusionMatrix(test$classe, pred_bst)    # confusion matrix
conf_bst

# C5.0 Rules
train_control <- trainControl(method = "cv", number = 10)
model_c5.0rules <- train(classe ~ ., data = train, trControl = train_control, method = "C5.0Rules")
model_c5.0rules
pred_c5.0rules <- predict(model_c5.0rules, test)                       # make prediction
conf_c5.0rules <- confusionMatrix(test$classe, pred_c5.0rules)    # confusion matrix
conf_c5.0rules

# C5.0 Tree
train_control <- trainControl(method = "cv", number = 10)
model_c5.0tree <- train(classe ~ ., data = train, trControl = train_control, method = "C5.0Tree")
model_c5.0tree
pred_c5.0tree <- predict(model_c5.0tree, test)                       # make prediction
conf_c5.0tree <- confusionMatrix(test$classe, pred_c5.0tree)    # confusion matrix
conf_c5.0tree

# conditional inference trees
# Ctree
train_control <- trainControl(method = "cv", number = 10)
model_ctree <- train(classe ~ ., data = train, trControl = train_control, method = "ctree")
model_ctree
pred_ctree <- predict(model_ctree, test)                       # make prediction
conf_ctree <- confusionMatrix(test$classe, pred_ctree)    # confusion matrix
conf_ctree

# Ctree2
train_control <- trainControl(method = "cv", number = 10)
model_ctree2 <- train(classe ~ ., data = train, trControl = train_control, method = "ctree2")
model_ctree2
pred_ctree2 <- predict(model_ctree2, test)                       # make prediction
conf_ctree2 <- confusionMatrix(test$classe, pred_ctree2)    # confusion matrix
conf_ctree2

# Random forest
model_rf <- randomForest(classe ~., train, ntree = 500)
model_rf
pred_rf <- predict(model_rf, test)                       # make prediction
conf_rf <- confusionMatrix(test$classe, pred_rf)    # confusion matrix
conf_rf

model <- c("model_tree", "model_cart", "model_cv", "model_c5.0 ", "model_bst",
           "model_c5.0rules", "model_c5.0tree", "model_ctree", "model_ctree2", "model_rf")

#------------------------------------------------------------------------------------------
# b. Verify model goodness of fit.

chisq.test(table(test$classe), prop.table(table(pred_tree)))       # pv = 0.2202
chisq.test(table(test$classe), prop.table(table(pred_cart)))       # pv = 0.2202
chisq.test(table(test$classe), prop.table(table(pred_cv)))         # pv = 0.2414 
chisq.test(table(test$classe), prop.table(table(pred_c5.0)))       # pv = 0.2202
chisq.test(table(test$classe), prop.table(table(pred_bst)))        # pv = 0.2650 
chisq.test(table(test$classe), prop.table(table(pred_c5.0rules)))  # pv = 0.2202
chisq.test(table(test$classe), prop.table(table(pred_c5.0tree)))   # pv = 0.2202
chisq.test(table(test$classe), prop.table(table(pred_ctree)))      # pv = 0.2202
chisq.test(table(test$classe), prop.table(table(pred_ctree2)))     # pv = 0.2202
chisq.test(table(test$classe), prop.table(table(pred_rf)))         # pv = 0.2202

conf_tree$overall[1]       
conf_cart$overall[1]       
conf_cv$overall[1]         
conf_c5.0$overall[1]       
conf_bst$overall[1]        
conf_c5.0rules$overall[1]  
conf_c5.0tree$overall[1]   
conf_ctree$overall[1]      
conf_ctree2$overall[1]    
conf_rf$overall[1]        

#-----------------------------------------------------------------------------------------
# c. Apply all the model validation techniques.

# 1
train_control <- trainControl(method = "cv", number = 10)
cvmodel1 <- train(classe ~ ., data = train, trControl = train_control, method = "rf") 
cvpred1 <- predict(cvmodel1, test)                        # make prediction
cvconf1 <- confusionMatrix(test$classe, pred_ctree)       # confusion matrix
cvconf1$overall[1]                                        # accuracy

# default
set.seed(123)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
rf_default <- train(classe ~ ., data = train, trControl = train_control, method = "rf",
                  metric = 'Accuracy', tuneGrid = expand.grid(.mtry = sqrt(ncol(train)))) 
pred_rf_default <- predict(rf_default, test)                            # make prediction
conf_rf_default <- confusionMatrix(test$classe, pred_rf_default)        # confusion matrix
conf_rf_default$overall[1]                                              # accuracy
varImp(rf_default)                                                      # var importance - 20

# random search for parameters
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = 'random')
rf_random <- train(classe ~ ., data = train, trControl = train_control, method = "rf",
                    metric = 'Accuracy', tuneLength = 15) 
pred_rf_random <- predict(rf_random, test)                            # make prediction
conf_rf_random <- confusionMatrix(test$classe, pred_rf_random)        # confusion matrix
conf_rf_random$overall[1]                                             # accuracy
varImp(rf_random)                                                     # var importance - 20

# Grid Search
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = 'grid')
rf_grid <- train(classe ~ ., data = train, trControl = train_control, method = "rf",
                   metric = 'Accuracy', tuneGrid = expand.grid(.mtry=c(1:15))) 
pred_rf_grid <- predict(rf_grid, test)                            # make prediction
conf_rf_grid <- confusionMatrix(test$classe, pred_rf_grid)        # confusion matrix
conf_rf_grid$overall[1]                                           # accuracy
varImp(rf_grid)                                                   # var importance - 20

# gradient boosting
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3, search = 'grid')
rf_gbm <- train(classe ~ ., data = train, trControl = train_control, method = "gbm",
                 metric = 'Accuracy') 
print(rf_gbm)
plot(rf_gbm)
pred_rf_gbm <- predict(rf_gbm, test)                             # make prediction
conf_rf_gbm <- confusionMatrix(test$classe, pred_rf_gbm)         # confusion matrix
conf_rf_gbm$overall[1]                                           # accuracy
summary(rf_gbm)                                                  # var importance - 18

# ---------------------------------------------------------------------------------------
# d. Make conclusions

# Problem was to predict how well the activity is performed
# The target variable is the 5 classe; 1 accurate and 4 type of error 
# occured during the activity

# error (target) detection was done by classifying an 
# execution to one of the mistake classes

# we could detect mistakes fairly accurately

# Gradient bossting model is most accurate with less number of predictors 
# Model is good fit and the Accuracy is 1

plot <- plot(conf_rf$table, col = topo.colors(6))

# -------------------------------------------------------------------------------------

