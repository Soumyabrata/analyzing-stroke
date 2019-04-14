# Load the different libraries
library(dplyr)
library(caret)
library(ggplot2)
library(imputeTS)
library(caTools)
library(e1071)
library(nnet)
library(rpart)
library(ggpubr)
library("randomForest")
library(factoextra)


# Load the dataset. The blank spaces are replaced by NA.
input_data <- read.csv("./dataset/train_2v.csv", na.strings=c("","NA"))

# Omit the rows that contain NA.
input_data<- na.omit(input_data)
colnames(input_data)

# The patient ID is not a feature, so we remove it. We also remove the stroke feature
input_features = input_data[,2:11]
data_without_id = input_data[,2:12]
head(input_features)

# Converting into factor type
input_features$gender <- as.factor(input_features$gender)
input_features$ever_married <- as.factor(input_features$ever_married)
input_features$work_type <- as.factor(input_features$work_type)
input_features$Residence_type <- as.factor(input_features$Residence_type)
input_features$smoking_status <- as.factor(input_features$smoking_status)

input_features[] <- data.matrix(input_features)


# Figure 1
res.pca <- prcomp(input_features, scale = TRUE)
fviz_eig(res.pca)


# Figure 2
fviz_pca_var(res.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)



# Section II (a): the pca values
std_dev <- res.pca[1]
sdev <- std_dev$sdev
eig_values <- sdev^2
pca_values <- eig_values/sum(eig_values)
pca_values <- pca_values*100
pca_values


# Figure 3 (a)
fviz_pca_ind(res.pca,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = "cos2", # Color by the quality of representation
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)



# Figure 3 (b)
groups <- as.factor(data_without_id$stroke)
fviz_pca_ind(res.pca,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = groups, # color by groups
             palette = c("#00AFBB", "#FC4E07"),
             
             ellipse.type = "confidence",
             legend.title = "Groups",
             repel = TRUE,
             addEllipses = TRUE # Concentration ellipses
)





# random downsampling
no_of_exps <-1000

minority_class <- data_without_id[data_without_id$stroke == 1,]
majority_class <- data_without_id[data_without_id$stroke == 0,]

# Neural network method

nnetwork_result = c()

for (i in 1:no_of_exps) {
  cat("Current experiment: ", i)
  
  
  majority_sample <- majority_class[sample(nrow(majority_class), 548), ]
  balanced_dataset <- rbind(minority_class, majority_sample)
  
  split = sample.split(balanced_dataset$stroke, SplitRatio = 0.70)
  train_set = subset(balanced_dataset, split == TRUE)
  test_set = subset(balanced_dataset, split == FALSE)
  
  nnetm <- train(as.factor(stroke) ~., data=train_set, method='nnet')
  print(nnetm)
  plot(nnetm)
  pred_nn <- predict(nnetm, newdata=test_set)
  
  # Output labels
  out_labels<-as.data.frame(test_set[, 11])
  out_labels<-t(out_labels)
  
  cm_nn = table(out_labels, pred_nn)
  
  #accuracy
  n_nn = sum(cm_nn)
  diag_nn = diag(cm_nn)
  accuracy_nn = sum(diag_nn) / n_nn
  accuracy_nn
  
  
  nnetwork_result[length(nnetwork_result)+1] = accuracy_nn
}

nnetwork_result





# Decision tree method

dtree_result = c()

for (i in 1:no_of_exps) {
  cat("Current experiment: ", i)
  
  
  majority_sample <- majority_class[sample(nrow(majority_class), 548), ]
  balanced_dataset <- rbind(minority_class, majority_sample)
  
  split = sample.split(balanced_dataset$stroke, SplitRatio = 0.70)
  train_set = subset(balanced_dataset, split == TRUE)
  test_set = subset(balanced_dataset, split == FALSE)
  

  train_set$stroke <- factor(train_set$stroke)
  classifier_dt = rpart(formula =stroke~ .,
                        data = train_set)
  print(classifier_dt)
  # Predicting the Test set results
  y_pred_dt= predict(classifier_dt, newdata = test_set, type = 'class')
  
  # Output labels
  out_labels<-as.data.frame(test_set[, 11])
  out_labels<-t(out_labels)
  
  # Making the Confusion Matrix
  cm_dt = table(out_labels, y_pred_dt)
  
  #accuracy
  n_dt = sum(cm_dt)
  diag_dt = diag(cm_dt)
  accuracy_dt = sum(diag_dt) / n_dt
  accuracy_dt

  dtree_result[length(dtree_result)+1] = accuracy_dt
}

dtree_result




# Random forest method


rforest_result = c()

for (i in 1:no_of_exps) {
  cat("Current experiment: ", i)
  
  majority_sample <- majority_class[sample(nrow(majority_class), 548), ]
  balanced_dataset <- rbind(minority_class, majority_sample)
  
  split = sample.split(balanced_dataset$stroke, SplitRatio = 0.70)
  trainData = subset(balanced_dataset, split == TRUE)
  testData = subset(balanced_dataset, split == FALSE)
  
  
  
  trainData$stroke <- as.character(trainData$stroke)
  trainData$stroke <- as.factor(trainData$stroke)
  stroke_rf = randomForest(stroke~., data=trainData, ntree=100, proximity=T)
  strokePred = predict(stroke_rf, newdata=testData)
  CM = table(strokePred, testData$stroke)
  accuracy = (sum(diag(CM)))/sum(CM)
  accuracy

  rforest_result[length(rforest_result)+1] = accuracy
}

rforest_result





nnetwork_result
dtree_result
rforest_result


x_name <- "Method"
y_name <- "Accuracy"

a1 <- replicate(no_of_exps, "Neural Network")
a2 <- replicate(no_of_exps, "Decision Tree")
a3 <- replicate(no_of_exps, "Random Forest")
a <- c(a1,a2,a3)

all_accuracies <- c(nnetwork_result,dtree_result,rforest_result)
df <- data.frame(a, all_accuracies)
colnames(df) <- c(x_name, y_name)
print(df)
head(df)


# Figure 4
ggdensity(df, x = "Accuracy",
          add = "mean", rug = TRUE,
          color = "Method", fill = "Method",
          palette = c("#0073C2FF", "#FC4E07", "#07fc9e"))


# saving the df
write.csv(df, file = "./results/my_results.csv")


# Table 1
cat("Mean accuracy of neural network: ", mean(nnetwork_result))
cat("Mean accuracy of decision tree: ", mean(dtree_result))
cat("Mean accuracy of random forest: ", mean(rforest_result))

