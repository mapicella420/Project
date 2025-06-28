install.packages("ggfortify")
install.packages("dplyr")



# ===============================================================
# 📦 LIBRARIES
# ===============================================================

# Data manipulation and visualization
library(readr)        # For reading CSV files
library(dplyr)        # For data wrangling
library(ggplot2)      # For general plotting
library(reshape2)     # For reshaping data (e.g., for heatmaps)
library(scales)       # For axis scaling

# Statistical and modeling tools
library(car)
library(stats)
library(glmnet)       # For ElasticNet regularization
library(pROC)         # For ROC and AUC
library(caret)        # For ML workflow (e.g., partitioning)

# Data formatting and utility
library(rlang)
library(tidyverse)
library(tidyr)
library(tibble)

# For correlation plot and PCA visualization
library(corrplot)
library(ggfortify)

# ML algorithms
library(e1071)        # For Naive Bayes, etc.

# ===============================================================
# 📁 DATA LOADING AND CLEANING
# ===============================================================

# Set working directory (adjust manually if needed)
setwd("/Users/salvatore/Desktop/HealtDataAnalytics/Classification")

# Load dataset
df <- read_csv("Prostate.csv")

view(df)

# Remove unneeded columns (e.g., 'samples', 'X', or 'index')
df <- dplyr::select(df, -matches("^samples$|^X$|^index$", ignore.case = TRUE))

# Dataset dimensions
cat("📦 Dataset dimensions: ", dim(df)[1], "rows ×", dim(df)[2], "columns\n")

# Convert response to factor
df$response <- as.factor(df$response)

# Check class distribution
table(df$response)
prop.table(table(df$response))

# Rename and clean target variable
df$target <- df$response
df$response <- NULL
levels(df$target)[levels(df$target) == "tumer"] <- "tumor"

# Confirm changes
table(df$target)

# ===============================================================
# 🔀 DATA SPLITTING AND SCALING
# ===============================================================

# Initial train-test split (80%-20%)
set.seed(42)
split_index <- createDataPartition(df$target, p = 0.8, list = FALSE)
train_df <- df[split_index, ]
test_df  <- df[-split_index, ]

# Standardize numeric features using training statistics
numeric_features <- names(train_df)[sapply(train_df, is.numeric) & names(train_df) != "target"]

# Compute training means and SDs
means <- sapply(train_df[, numeric_features], mean)
sds   <- sapply(train_df[, numeric_features], sd)

# Apply scaling
train_scaled <- train_df
train_scaled[, numeric_features] <- scale(train_df[, numeric_features], center = means, scale = sds)

test_scaled <- test_df
test_scaled[, numeric_features] <- scale(test_df[, numeric_features], center = means, scale = sds)

# Print scaled feature statistics
cat("📊 Scaled feature means (train):\n")
print(round(colMeans(train_scaled[, numeric_features]), 4))

cat("\n📊 Scaled feature SDs (train):\n")
print(round(apply(train_scaled[, numeric_features], 2, sd), 4))

cat("\n📊 Scaled feature means (test):\n")
print(round(colMeans(test_scaled[, numeric_features]), 4))

cat("\n📊 Scaled feature SDs (test):\n")
print(round(apply(test_scaled[, numeric_features], 2, sd), 4))
# ===============================================================
# 🔄 ELASTICNET CLASSIFIER WITHOUT FEATURE SELECTION
# ===============================================================

# Prepare data matrices
X_train <- as.matrix(train_scaled %>% select(-target))
y_train <- train_scaled$target
X_test  <- as.matrix(test_scaled %>% select(-target))
y_test  <- test_scaled$target

# Cross-validation over different alpha values [0, 1]
set.seed(42)
alphas <- seq(0, 1, by = 0.1)
cv_results <- list()
auc_scores <- c()

for (a in alphas) {
  cv_model <- cv.glmnet(X_train, y_train, alpha = a, family = "binomial",
                        type.measure = "auc", nfolds = 10)
  cv_results[[as.character(a)]] <- cv_model
  auc_scores <- c(auc_scores, max(cv_model$cvm))
}

# Plot AUC vs Alpha
df_auc <- data.frame(alpha = alphas, AUC = auc_scores)
best_alpha <- df_auc$alpha[which.max(df_auc$AUC)]

ggplot(df_auc, aes(x = alpha, y = AUC)) +
  geom_line(color = "#1f77b4", linewidth = 1.2) +
  geom_point(size = 3, color = ifelse(df_auc$alpha == best_alpha, "#d62728", "#1f77b4")) +
  geom_text(aes(label = ifelse(alpha == best_alpha, paste0("Best α = ", alpha), "")),
            vjust = -1.2, color = "#d62728", size = 5) +
  theme_minimal(base_size = 14) +
  labs(title = "AUC vs Alpha (ElasticNet)",
       x = expression(alpha), y = "Cross-Validated AUC")

# AUC vs log(lambda)
best_model <- cv_results[[as.character(best_alpha)]]
lambda_df <- data.frame(lambda = best_model$lambda, AUC = best_model$cvm, SD = best_model$cvsd)

ggplot(lambda_df, aes(x = log(lambda), y = AUC)) +
  geom_line(color = "#2ca02c", linewidth = 1.2) +
  geom_ribbon(aes(ymin = AUC - SD, ymax = AUC + SD), alpha = 0.2, fill = "#2ca02c") +
  geom_vline(xintercept = log(best_model$lambda.min), linetype = "dashed", color = "#d62728") +
  annotate("text", x = log(best_model$lambda.min), y = max(lambda_df$AUC),
           label = paste0("Best λ = ", round(best_model$lambda.min, 4)),
           hjust = -0.1, color = "#d62728", size = 5) +
  theme_minimal(base_size = 14) +
  labs(title = paste("AUC vs log(Lambda) – α =", best_alpha),
       x = "log(Lambda)", y = "Cross-Validated AUC")

# Train final ElasticNet model
final_model <- glmnet(X_train, y_train, alpha = best_alpha,
                      lambda = best_model$lambda.min, family = "binomial")

# ===============================================================
# 🎯 PERFORMANCE METRICS
# ===============================================================

# Predict and evaluate on train set
train_probs <- predict(final_model, X_train, type = "response")
train_preds <- factor(ifelse(train_probs > 0.5, "tumor", "normal"), levels = c("normal", "tumor"))
cat("📦 ElasticNet – Training Set Performance:\n")
print(confusionMatrix(train_preds, y_train))

# Predict and evaluate on test set
test_probs <- predict(final_model, X_test, type = "response")
test_preds <- factor(ifelse(test_probs > 0.5, "tumor", "normal"), levels = c("normal", "tumor"))
conf_test <- confusionMatrix(test_preds, y_test)
cat("📦 ElasticNet – Test Set Performance:\n")
print(conf_test)

# Precision, Recall, F1
cm <- conf_test$table
TP <- cm[2, 2]; FP <- cm[1, 2]; FN <- cm[2, 1]
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("\n📈 TEST Metrics:\n")
cat("   ➤ Precision:", round(precision, 3), "\n")
cat("   ➤ Recall:   ", round(recall, 3), "\n")
cat("   ➤ F1-score: ", round(f1_score, 3), "\n")

# ===============================================================
# 📈 ROC CURVE – TEST SET
# ===============================================================

roc_obj <- roc(response = as.numeric(y_test) == 2,
               predictor = as.numeric(test_probs),
               direction = "<", quiet = TRUE)
auc_val <- auc(roc_obj)

roc_df_en <- data.frame(
  FPR = rev(1 - roc_obj$specificities),
  TPR = rev(roc_obj$sensitivities)
)
roc_df_en <- rbind(c(0, 0), roc_df_en, c(1, 1))
colnames(roc_df_en) <- c("FPR", "TPR")

ggplot(roc_df_en, aes(x = FPR, y = TPR)) +
  geom_area(fill = "lightgreen", alpha = 0.2) +
  geom_line(color = "lightgreen", linewidth = 1.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  annotate("text", x = 0.7, y = 0.15,
           label = paste("AUC =", round(auc_val, 2)),
           size = 6, fontface = "italic", color = "black") +
  labs(title = "ROC Curve – Test Set",
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
  theme_minimal(base_size = 16)

# ===============================================================
# 🔍 PCA VISUALIZATION + DECISION BOUNDARY
# ===============================================================

# PCA on training set
pca_en <- prcomp(train_scaled %>% select(-target), center = TRUE, scale. = TRUE)
train_pca_en <- as.data.frame(pca_en$x[, 1:2])
train_pca_en$target <- train_scaled$target

# Logistic regression on PC1 and PC2 (for visualization)
en_vis_model <- glm(target ~ ., data = train_pca_en, family = "binomial")

# Grid for decision boundary
grid_df <- expand.grid(
  PC1 = seq(min(train_pca_en$PC1) - 1, max(train_pca_en$PC1) + 1, length.out = 200),
  PC2 = seq(min(train_pca_en$PC2) - 1, max(train_pca_en$PC2) + 1, length.out = 200)
)
grid_df$prob <- predict(en_vis_model, newdata = grid_df, type = "response")
grid_df$prediction <- factor(ifelse(grid_df$prob > 0.5, "tumor", "normal"),
                             levels = c("normal", "tumor"))

# Explained variance
explained_var <- round(100 * (pca_en$sdev^2 / sum(pca_en$sdev^2)), 1)
x_lab <- paste0("PC1 (", explained_var[1], "%)")
y_lab <- paste0("PC2 (", explained_var[2], "%)")

# 📊 TRAIN SET
ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = train_pca_en, aes(x = PC1, y = PC2, color = target), size = 2.5) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(
    title = "ElasticNet Logistic – Decision Boundary (Train Set)",
    subtitle = "Background: Prediction – Points: Actual Class",
    x = x_lab, y = y_lab
  ) +
  theme_minimal(base_size = 14)

# 📊 TEST SET
test_pca_en <- predict(pca_en, newdata = test_scaled %>% select(-target))
test_pca_en <- as.data.frame(test_pca_en[, 1:2])
test_pca_en$target <- test_scaled$target

ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = test_pca_en, aes(x = PC1, y = PC2, color = target), size = 2.5) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(
    title = "ElasticNet Logistic – Decision Boundary (Test Set)",
    subtitle = "Background: Prediction – Points: Actual Class",
    x = x_lab, y = y_lab
  ) +
  theme_minimal(base_size = 14)




# ===============================================================
# RIDGE LOGISTIC REGRESSION (α = 0) – WITHOUT FEATURE SELECTION
# ===============================================================

# 1. Prepare Training and Test Sets
X_train <- as.matrix(train_scaled %>% select(-target))
y_train <- train_scaled$target
X_test  <- as.matrix(test_scaled %>% select(-target))
y_test  <- test_scaled$target

# 2. Cross-Validation Setup (10-fold)
set.seed(42)
cv_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE,
  verboseIter = TRUE
)

# 3. Define Lambda Grid for Ridge (α = 0)
ridge_grid <- expand.grid(
  alpha = 0,
  lambda = exp(seq(log(1e-5), log(10), length.out = 100))
)

# 4. Train Ridge Model on Training Set
ridge_model <- train(
  x = X_train,
  y = y_train,
  method = "glmnet",
  tuneGrid = ridge_grid,
  trControl = cv_control,
  metric = "ROC"
)

# ✅ Extract best lambda
best_lambda <- ridge_model$bestTune$lambda

# 5. AUC vs log(Lambda) Plot – styled to match ElasticNet
ridge_results <- ridge_model$results
ridge_df <- ridge_results[ridge_results$alpha == 0, c("lambda", "ROC", "ROCSD")]
colnames(ridge_df) <- c("lambda", "AUC", "SD")

ggplot(ridge_df, aes(x = log(lambda), y = AUC)) +
  geom_line(color = "#2ca02c", linewidth = 1.2) +
  geom_ribbon(aes(ymin = AUC - SD, ymax = AUC + SD), alpha = 0.2, fill = "#2ca02c") +
  geom_vline(xintercept = log(best_lambda), linetype = "dashed", color = "#d62728") +
  annotate("text",
           x = log(best_lambda),
           y = max(ridge_df$AUC),
           label = paste0("Best λ = ", signif(best_lambda, 4)),
           hjust = -0.1, color = "#d62728", size = 3) +
  theme_minimal(base_size = 14) +
  labs(title = "AUC vs log(Lambda) – Ridge Regression",
       x = "log(Lambda)",
       y = "Cross-Validated AUC")

# 6. Model Performance on Training Set
train_probs <- predict(ridge_model$finalModel, X_train, s = best_lambda, type = "response")
train_preds <- factor(ifelse(train_probs > 0.5, "tumor", "normal"), levels = c("normal", "tumor"))
cat("📦 Ridge – Training Set Performance:\n")
print(confusionMatrix(train_preds, y_train))
cat("F1 Score (TRAIN):", round(F_meas(train_preds, y_train, relevant = "tumor"), 3), "\n")

# 7. Model Performance on Test Set
test_probs <- predict(ridge_model$finalModel, X_test, s = best_lambda, type = "response")
test_preds <- factor(ifelse(test_probs > 0.5, "tumor", "normal"), levels = c("normal", "tumor"))
cat("📦 Ridge – Test Set Performance:\n")
print(confusionMatrix(test_preds, y_test))
cat("F1 Score (TEST):", round(F_meas(test_preds, y_test, relevant = "tumor"), 3), "\n")

# 8. ROC Curve – Test Set
roc_obj <- roc(response = as.numeric(y_test) == 2,
               predictor = as.numeric(test_probs),
               direction = "<",
               quiet = TRUE)
auc_val <- auc(roc_obj)
roc_df <- data.frame(
  FPR = rev(1 - roc_obj$specificities),
  TPR = rev(roc_obj$sensitivities)
)
roc_df <- rbind(c(0, 0), roc_df, c(1, 1))
colnames(roc_df) <- c("FPR", "TPR")

ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_area(fill = "lightblue", alpha = 0.2) +
  geom_line(color = "#1f77b4", linewidth = 1.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  annotate("text", x = 0.7, y = 0.15,
           label = paste("AUC =", round(auc_val, 2)),
           size = 6, fontface = "italic", color = "black") +
  labs(title = "ROC Curve – Ridge Regression (Test Set)",
       x = "False Positive Rate (FPR)",
       y = "True Positive Rate (TPR)") +
  theme_minimal(base_size = 16) +
  coord_fixed()

# ===============================================================
# PCA VISUALIZATION – RIDGE (Train and Test Sets)
# ===============================================================

# PCA on training set
pca_ridge <- prcomp(train_scaled %>% select(-target), center = TRUE, scale. = TRUE)
train_pca_ridge <- as.data.frame(pca_ridge$x[, 1:2])
train_pca_ridge$target <- train_scaled$target

# Logistic regression on PC1 and PC2
ridge_vis_model <- glm(target ~ ., data = train_pca_ridge, family = "binomial")

# Grid for decision boundary
grid_df <- expand.grid(
  PC1 = seq(min(train_pca_ridge$PC1) - 1, max(train_pca_ridge$PC1) + 1, length.out = 200),
  PC2 = seq(min(train_pca_ridge$PC2) - 1, max(train_pca_ridge$PC2) + 1, length.out = 200)
)
grid_df$prob <- predict(ridge_vis_model, newdata = grid_df, type = "response")
grid_df$prediction <- factor(ifelse(grid_df$prob > 0.5, "tumor", "normal"),
                             levels = c("normal", "tumor"))

# Variance explained by PC1 and PC2
explained_var_ridge <- round(100 * (pca_ridge$sdev^2 / sum(pca_ridge$sdev^2)), 1)
x_lab_ridge <- paste0("PC1 (", explained_var_ridge[1], "%)")
y_lab_ridge <- paste0("PC2 (", explained_var_ridge[2], "%)")

# Plot – Train Set
ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = train_pca_ridge, aes(x = PC1, y = PC2, color = target), size = 2.5) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(
    title = "Ridge Logistic – Decision Boundary (Train Set)",
    subtitle = "Background: prediction – Points: true class",
    x = x_lab_ridge,
    y = y_lab_ridge
  ) +
  theme_minimal(base_size = 14)

# PCA on test set
test_pca_ridge <- predict(pca_ridge, newdata = test_scaled %>% select(-target))
test_pca_ridge <- as.data.frame(test_pca_ridge[, 1:2])
test_pca_ridge$target <- test_scaled$target

ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = test_pca_ridge, aes(x = PC1, y = PC2, color = target), size = 2.5) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(
    title = "Ridge Logistic – Decision Boundary (Test Set)",
    subtitle = "Background: prediction – Points: true class",
    x = x_lab_ridge,
    y = y_lab_ridge
  ) +
  theme_minimal(base_size = 14)






# ===============================================================
# LASSO LOGISTIC REGRESSION (α = 1) – WITHOUT FEATURE SELECTION
# ===============================================================

# 1. Prepare Training and Test Sets
X_train <- as.matrix(train_scaled %>% select(-target))
y_train <- train_scaled$target
X_test  <- as.matrix(test_scaled %>% select(-target))
y_test  <- test_scaled$target

# 2. Cross-validation to identify best lambda
set.seed(42)
lasso_cv <- cv.glmnet(
  X_train, y_train,
  alpha = 1,
  family = "binomial",
  type.measure = "auc",
  nfolds = 10
)

# 3. AUC vs log(lambda) Plot
lambda_df <- data.frame(
  lambda = lasso_cv$lambda,
  AUC = lasso_cv$cvm,
  SD = lasso_cv$cvsd
)

ggplot(lambda_df, aes(x = log(lambda), y = AUC)) +
  geom_line(color = "#ff7f0e", linewidth = 1.2) +
  geom_ribbon(aes(ymin = AUC - SD, ymax = AUC + SD), alpha = 0.2, fill = "#ff7f0e") +
  geom_vline(xintercept = log(lasso_cv$lambda.min), linetype = "dashed", color = "#d62728") +
  annotate("text", x = log(lasso_cv$lambda.min), y = max(lambda_df$AUC),
           label = paste0("Best λ = ", round(lasso_cv$lambda.min, 4)),
           hjust = -0.1, color = "#d62728", size = 5) +
  theme_minimal(base_size = 14) +
  labs(title = "AUC vs log(Lambda) – Lasso (α = 1)",
       x = "log(Lambda)", y = "Cross-Validated AUC")

# 4. Final Lasso Model
lasso_model <- glmnet(
  X_train, y_train,
  alpha = 1,
  lambda = lasso_cv$lambda.min,
  family = "binomial"
)

# 5. Performance on TRAIN Set
train_probs_lasso <- predict(lasso_model, X_train, type = "response")
train_preds_lasso <- factor(ifelse(train_probs_lasso > 0.5, "tumor", "normal"),
                            levels = c("normal", "tumor"))
conf_train <- confusionMatrix(train_preds_lasso, y_train)
cat("📦 Lasso – Training Set Performance:\n"); print(conf_train)

# 6. Performance on TEST Set
test_probs_lasso <- predict(lasso_model, X_test, type = "response")
test_preds_lasso <- factor(ifelse(test_probs_lasso > 0.5, "tumor", "normal"),
                           levels = c("normal", "tumor"))
conf_test <- confusionMatrix(test_preds_lasso, y_test)
cat("📦 Lasso – Test Set Performance:\n"); print(conf_test)

# 7. Precision, Recall, F1-Score
cm <- conf_test$table
TP <- cm[2, 2]; FP <- cm[1, 2]; FN <- cm[2, 1]
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("\n📈 TEST Metrics:\n")
cat("   ➤ Precision:", round(precision, 3), "\n")
cat("   ➤ Recall:   ", round(recall, 3), "\n")
cat("   ➤ F1-score: ", round(f1_score, 3), "\n")

# 8. ROC Curve – Test Set
roc_lasso <- roc(response = as.numeric(y_test) == 2,
                 predictor = as.numeric(test_probs_lasso),
                 direction = "<", quiet = TRUE)
auc_lasso <- auc(roc_lasso)

roc_df_lasso <- data.frame(
  FPR = rev(1 - roc_lasso$specificities),
  TPR = rev(roc_lasso$sensitivities)
)
roc_df_lasso <- rbind(c(0, 0), roc_df_lasso, c(1, 1))
colnames(roc_df_lasso) <- c("FPR", "TPR")

ggplot(roc_df_lasso, aes(x = FPR, y = TPR)) +
  geom_area(fill = "#ff7f0e", alpha = 0.2) +
  geom_line(color = "#ff7f0e", linewidth = 1.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  annotate("text", x = 0.7, y = 0.15,
           label = paste("AUC =", round(auc_lasso, 2)),
           size = 6, fontface = "italic", color = "black") +
  labs(title = "ROC Curve – Lasso (Test Set)",
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
  theme_minimal(base_size = 16)

# ===============================================================
# PCA VISUALIZATION – LASSO (Train and Test Sets)
# ===============================================================

# PCA on training set
pca_lasso <- prcomp(train_scaled %>% select(-target), center = TRUE, scale. = TRUE)
train_pca_lasso <- as.data.frame(pca_lasso$x[, 1:2])
train_pca_lasso$target <- train_scaled$target

# Logistic model on PC1 and PC2 (for visualization only)
lasso_vis_model <- glm(target ~ ., data = train_pca_lasso, family = "binomial")

# Create grid for decision boundary
grid_df <- expand.grid(
  PC1 = seq(min(train_pca_lasso$PC1) - 1, max(train_pca_lasso$PC1) + 1, length.out = 200),
  PC2 = seq(min(train_pca_lasso$PC2) - 1, max(train_pca_lasso$PC2) + 1, length.out = 200)
)
grid_df$prob <- predict(lasso_vis_model, newdata = grid_df, type = "response")
grid_df$prediction <- factor(ifelse(grid_df$prob > 0.5, "tumor", "normal"),
                             levels = c("normal", "tumor"))


# Calcola la varianza spiegata dalle prime due componenti
explained_var_lasso <- round(100 * (pca_lasso$sdev^2 / sum(pca_lasso$sdev^2)), 1)
x_lab_lasso <- paste0("PC1 (", explained_var_lasso[1], "%)")
y_lab_lasso <- paste0("PC2 (", explained_var_lasso[2], "%)")

# Plot – Decision Boundary on Training Set
ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = train_pca_lasso, aes(x = PC1, y = PC2, color = target), size = 2.5) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(
    title = "Lasso Logistic – Decision Boundary (Train Set)",
    subtitle = "Background: prediction – Points: true class",
    x = x_lab_lasso,
    y = y_lab_lasso
  ) +
  theme_minimal(base_size = 14)


# PCA on test set
pca_test_lasso <- predict(pca_lasso, newdata = test_scaled %>% select(-target))
test_pca_lasso <- as.data.frame(pca_test_lasso[, 1:2])
test_pca_lasso$target <- test_scaled$target

# Plot – Decision Boundary on Test Set
ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = test_pca_lasso, aes(x = PC1, y = PC2, color = target), size = 2.5) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(
    title = "Lasso Logistic – Decision Boundary (Test Set)",
    subtitle = "Background: prediction – Points: true class",
    x = x_lab_lasso,
    y = y_lab_lasso
  ) +
  theme_minimal(base_size = 14)





# ================================
# Feature selection using Lasso
# ================================

# 1. Prepare the data
X_train <- as.matrix(train_scaled %>% select(-target))
y_train <- train_scaled$target

# 2. Lasso with cross-validation (for feature selection only)
set.seed(42)
lasso_fs <- cv.glmnet(
  X_train, y_train,
  alpha = 1,            # Pure Lasso
  family = "binomial",
  type.measure = "auc"
)

# 3. Extract coefficients at lambda.min
lasso_coef <- coef(lasso_fs, s = "lambda.min")

# 4. Get feature names with non-zero coefficients (excluding the intercept)
selected_features <- rownames(lasso_coef)[which(lasso_coef != 0)][-1]  # remove "(Intercept)"

# 5. Display selected features
cat("Selected features from Lasso (alpha = 1):\n")
print(selected_features)




# ================================
# ElasticNet Logistic Regression (selected features only)
# ================================

# 1. Prepare the data using only selected features
X_train_fs <- as.matrix(train_scaled %>% select(all_of(selected_features)))
X_test_fs  <- as.matrix(test_scaled  %>% select(all_of(selected_features)))

y_train <- train_scaled$target
y_test  <- test_scaled$target

# 2. Cross-validation over alpha ∈ [0, 1]
set.seed(42)
alphas <- seq(0, 1, by = 0.1)
cv_results_fs <- list()
auc_scores_fs <- c()

for (a in alphas) {
  model <- cv.glmnet(
    X_train_fs, y_train,
    alpha = a,
    family = "binomial",
    type.measure = "auc",
    nfolds = 10
  )
  cv_results_fs[[as.character(a)]] <- model
  auc_scores_fs <- c(auc_scores_fs, max(model$cvm))
}

# 3. Plot AUC vs Alpha
df_auc_fs <- data.frame(alpha = alphas, AUC = auc_scores_fs)
best_alpha_fs <- df_auc_fs$alpha[which.max(df_auc_fs$AUC)]

ggplot(df_auc_fs, aes(x = alpha, y = AUC)) +
  geom_line(color = "#2ca02c", linewidth = 1.2) +
  geom_point(size = 3, color = ifelse(df_auc_fs$alpha == best_alpha_fs, "#d62728", "#2ca02c")) +
  geom_text(aes(label = ifelse(alpha == best_alpha_fs, paste0("Best α = ", alpha), "")),
            vjust = -1.2, color = "#d62728", size = 3) +
  theme_minimal(base_size = 14) +
  labs(title = "AUC vs Alpha (ElasticNet on Selected Features)",
       x = expression(alpha), y = "Cross-Validated AUC")

# 4. Plot AUC vs log(Lambda)
best_model_fs <- cv_results_fs[[as.character(best_alpha_fs)]]

lambda_df_fs <- data.frame(
  lambda = best_model_fs$lambda,
  AUC = best_model_fs$cvm,
  SD = best_model_fs$cvsd
)

ggplot(lambda_df_fs, aes(x = log(lambda), y = AUC)) +
  geom_line(color = "#2ca02c", linewidth = 1.2) +
  geom_ribbon(aes(ymin = AUC - SD, ymax = AUC + SD), alpha = 0.2, fill = "#2ca02c") +
  geom_vline(xintercept = log(best_model_fs$lambda.min), linetype = "dashed", color = "#d62728") +
  annotate("text", x = log(best_model_fs$lambda.min), y = max(lambda_df_fs$AUC),
           label = paste0("Best λ = ", round(best_model_fs$lambda.min, 4)),
           hjust = -0.1, color = "#d62728", size = 3) +
  theme_minimal(base_size = 14) +
  labs(title = paste("AUC vs log(Lambda) – α =", best_alpha_fs),
       x = "log(Lambda)", y = "Cross-Validated AUC")

# ============================================
# Final ElasticNet model training
# ============================================

final_model_fs <- glmnet(
  X_train_fs, y_train,
  alpha = best_alpha_fs,
  lambda = best_model_fs$lambda.min,
  family = "binomial"
)

# ============================================
# Prediction on TRAIN set
# ============================================
train_probs_fs <- predict(final_model_fs, X_train_fs, type = "response")
train_preds_fs <- factor(ifelse(train_probs_fs > 0.5, "tumor", "normal"),
                         levels = c("normal", "tumor"))

conf_train <- confusionMatrix(train_preds_fs, y_train)
cat("TRAIN Performance:\n")
print(conf_train)

# F1 on train
cm_train <- conf_train$table
TP <- cm_train[2, 2]; FP <- cm_train[1, 2]
FN <- cm_train[2, 1]; TN <- cm_train[1, 1]
precision_train <- TP / (TP + FP)
recall_train    <- TP / (TP + FN)
f1_train        <- 2 * precision_train * recall_train / (precision_train + recall_train)

cat("\nTRAIN Metrics:\n")
cat("   Precision:", round(precision_train, 3), "\n")
cat("   Recall:   ", round(recall_train, 3), "\n")
cat("   F1-score: ", round(f1_train, 3), "\n")

# ============================================
# Prediction on TEST set
# ============================================

test_probs_fs <- predict(final_model_fs, X_test_fs, type = "response")
test_preds_fs <- factor(ifelse(test_probs_fs > 0.5, "tumor", "normal"),
                        levels = c("normal", "tumor"))

conf_test <- confusionMatrix(test_preds_fs, y_test)
cat("\nTEST Performance:\n")
print(conf_test)

# F1 on test
cm <- conf_test$table
TP <- cm[2, 2]; FP <- cm[1, 2]
FN <- cm[2, 1]; TN <- cm[1, 1]
precision <- TP / (TP + FP)
recall    <- TP / (TP + FN)
f1_score  <- 2 * precision * recall / (precision + recall)

cat("\nTEST Metrics:\n")
cat("   Precision:", round(precision, 3), "\n")
cat("   Recall:   ", round(recall, 3), "\n")
cat("   F1-score: ", round(f1_score, 3), "\n")

# ============================================
# ROC Curve + AUC
# ============================================

roc_fs <- roc(response = as.numeric(y_test) == 2,
              predictor = as.numeric(test_probs_fs),
              direction = "<", quiet = TRUE)

auc_fs <- auc(roc_fs)
roc_df_fs <- data.frame(
  FPR = rev(1 - roc_fs$specificities),
  TPR = rev(roc_fs$sensitivities)
)
roc_en_fs <- rbind(c(0, 0), roc_df_fs, c(1, 1))
colnames(roc_en_fs) <- c("FPR", "TPR")

ggplot(roc_en_fs, aes(x = FPR, y = TPR)) +
  geom_area(fill = "#2ca02c", alpha = 0.2) +
  geom_line(color = "#2ca02c", linewidth = 1.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  annotate("text", x = 0.7, y = 0.15,
           label = paste("AUC =", round(auc_fs, 3)),
           size = 6, fontface = "italic", color = "black") +
  labs(title = "ROC Curve – ElasticNet (Selected Features)",
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
  theme_minimal(base_size = 16)

# Coefficients of final ElasticNet model
enet_coef <- coef(final_model_fs)

# Remove intercept and filter non-zero coefficients
enet_selected_features <- rownames(enet_coef)[which(enet_coef != 0)][-1]

cat("ElasticNet retained", length(enet_selected_features), "features (out of", length(selected_features), "from Lasso):\n")
print(enet_selected_features)

# Optionally: which features were removed
excluded_by_enet <- setdiff(selected_features, enet_selected_features)
cat("\nFeatures excluded by ElasticNet:\n")
print(excluded_by_enet)

# ============================
# PCA for ElasticNet – Train Set
# ============================

pca_enet <- prcomp(train_scaled %>% select(all_of(selected_features)),
                   center = TRUE, scale. = TRUE)

train_pca_enet <- as.data.frame(pca_enet$x[, 1:2])
train_pca_enet$target <- train_scaled$target

# ============================
# Logistic model on PC1-PC2 (for boundary only)
# ============================
enet_pca_model <- glm(target ~ ., data = train_pca_enet, family = "binomial")

# Grid for decision boundary
x_min <- min(train_pca_enet$PC1) - 1
x_max <- max(train_pca_enet$PC1) + 1
y_min <- min(train_pca_enet$PC2) - 1
y_max <- max(train_pca_enet$PC2) + 1

grid_df <- expand.grid(
  PC1 = seq(x_min, x_max, length.out = 200),
  PC2 = seq(y_min, y_max, length.out = 200)
)

grid_df$prob <- predict(enet_pca_model, newdata = grid_df, type = "response")
grid_df$prediction <- factor(ifelse(grid_df$prob > 0.5, "tumor", "normal"),
                             levels = c("normal", "tumor"))

# ============================
# Get PCA explained variance
# ============================
expl_var_enet <- round(summary(pca_enet)$importance[2, 1:2] * 100, 1)
pc1_lab <- paste0("PC1 (", expl_var_enet[1], "%)")
pc2_lab <- paste0("PC2 (", expl_var_enet[2], "%)")

# ============================
# Plot – Train set decision boundary
# ============================
ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = train_pca_enet, aes(x = PC1, y = PC2, color = target), size = 2.5, alpha = 0.9) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(title = "ElasticNet – Decision Boundary (Train Set)",
       subtitle = "Background = prediction, Point = true class",
       x = pc1_lab, y = pc2_lab) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")

# ============================
# PCA on Test Set
# ============================
pca_test_enet <- predict(pca_enet, newdata = test_scaled %>% select(all_of(selected_features)))
test_pca_enet <- as.data.frame(pca_test_enet[, 1:2])
test_pca_enet$target <- test_scaled$target

# ============================
# Plot – Test set decision boundary
# ============================
ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = test_pca_enet, aes(x = PC1, y = PC2, color = target), size = 2.5, alpha = 0.9) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(title = "ElasticNet – Decision Boundary (Test Set)",
       subtitle = "Background = prediction, Point = true class",
       x = pc1_lab, y = pc2_lab) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")







# ========================================================
# RIDGE Logistic Regression – Using Lasso-Selected Features
# ========================================================

# 1. Prepare the data
X_train_ridge <- as.matrix(train_scaled %>% select(all_of(selected_features)))
X_test_ridge  <- as.matrix(test_scaled  %>% select(all_of(selected_features)))

y_train <- train_scaled$target
y_test  <- test_scaled$target

# 2. Cross-validation for lambda (alpha = 0 → Ridge)
set.seed(42)
ridge_cv <- cv.glmnet(
  X_train_ridge, y_train,
  alpha = 0,
  family = "binomial",
  type.measure = "auc",
  nfolds = 10
)

# 3. Plot AUC vs log(lambda)
lambda_df_ridge <- data.frame(
  lambda = ridge_cv$lambda,
  AUC = ridge_cv$cvm,
  SD = ridge_cv$cvsd
)

ggplot(lambda_df_ridge, aes(x = log(lambda), y = AUC)) +
  geom_line(color = "#1f77b4", linewidth = 1.2) +
  geom_ribbon(aes(ymin = AUC - SD, ymax = AUC + SD), alpha = 0.2, fill = "#1f77b4") +
  geom_vline(xintercept = log(ridge_cv$lambda.min), linetype = "dashed", color = "#d62728") +
  annotate("text", x = log(ridge_cv$lambda.min), y = max(lambda_df_ridge$AUC),
           label = paste0("Best λ = ", round(ridge_cv$lambda.min, 4)),
           hjust = -0.1, color = "#d62728", size = 5) +
  theme_minimal(base_size = 14) +
  labs(title = "AUC vs log(Lambda) – Ridge (α = 0)",
       x = "log(Lambda)", y = "Cross-Validated AUC")

# 4. Fit the final model using lambda.min
ridge_model <- glmnet(
  X_train_ridge, y_train,
  alpha = 0,
  lambda = ridge_cv$lambda.min,
  family = "binomial"
)

# 5. Predictions on training set
train_probs_ridge <- predict(ridge_model, X_train_ridge, type = "response")
train_preds_ridge <- factor(ifelse(train_probs_ridge > 0.5, "tumor", "normal"),
                            levels = c("normal", "tumor"))
conf_train_ridge <- confusionMatrix(train_preds_ridge, y_train)

# Compute F1 score (train)
cm_train <- conf_train_ridge$table
TP <- cm_train[2, 2]; FP <- cm_train[1, 2]
FN <- cm_train[2, 1]; TN <- cm_train[1, 1]
precision_train <- TP / (TP + FP)
recall_train <- TP / (TP + FN)
f1_train <- 2 * precision_train * recall_train / (precision_train + recall_train)

cat("Ridge – TRAIN performance:\n")
print(conf_train_ridge)
cat("\nTRAIN metrics:\n")
cat("   Precision:", round(precision_train, 3), "\n")
cat("   Recall:   ", round(recall_train, 3), "\n")
cat("   F1-score: ", round(f1_train, 3), "\n")

# 6. Predictions on test set
test_probs_ridge <- predict(ridge_model, X_test_ridge, type = "response")
test_preds_ridge <- factor(ifelse(test_probs_ridge > 0.5, "tumor", "normal"),
                           levels = c("normal", "tumor"))
conf_test_ridge <- confusionMatrix(test_preds_ridge, y_test)

# Compute F1 score (test)
cm_test <- conf_test_ridge$table
TP <- cm_test[2, 2]; FP <- cm_test[1, 2]
FN <- cm_test[2, 1]; TN <- cm_test[1, 1]
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * precision * recall / (precision + recall)

cat("\nRidge – TEST performance:\n")
print(conf_test_ridge)
cat("\nTEST metrics:\n")
cat("   Precision:", round(precision, 3), "\n")
cat("   Recall:   ", round(recall, 3), "\n")
cat("   F1-score: ", round(f1_score, 3), "\n")

# 7. ROC curve and AUC
roc_ridge_fs <- roc(response = as.numeric(y_test) == 2,
                 predictor = as.numeric(test_probs_ridge),
                 direction = "<", quiet = TRUE)
auc_ridge <- auc(roc_ridge)

roc_ridge_fs <- data.frame(
  FPR = rev(1 - roc_ridge_fs$specificities),
  TPR = rev(roc_ridge_fs$sensitivities)
)
roc_ridge_fs <- rbind(c(0, 0), roc_ridge_fs, c(1, 1))
colnames(roc_ridge_fs) <- c("FPR", "TPR")

ggplot(roc_ridge_fs, aes(x = FPR, y = TPR)) +
  geom_area(fill = "#1f77b4", alpha = 0.2) +
  geom_line(color = "#1f77b4", linewidth = 1.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  annotate("text", x = 0.7, y = 0.15,
           label = paste("AUC =", round(auc_ridge, 3)),
           size = 6, fontface = "italic", color = "black") +
  labs(title = "ROC Curve – Ridge (Feature Selection)",
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
  theme_minimal(base_size = 16)

# 8. Coefficients used in the final Ridge model
ridge_coef <- coef(ridge_model)
ridge_selected_features <- rownames(ridge_coef)[which(ridge_coef != 0)][-1]

cat("\nRidge used", length(ridge_selected_features), "features (out of", length(selected_features), "from Lasso):\n")
print(ridge_selected_features)

excluded_by_ridge <- setdiff(selected_features, ridge_selected_features)
cat("\nFeatures eliminated by Ridge:\n")
print(excluded_by_ridge)

# ============================
# PCA for Ridge – Train Set
# ============================

pca_ridge <- prcomp(train_scaled %>% select(all_of(selected_features)),
                    center = TRUE, scale. = TRUE)

train_pca_ridge <- as.data.frame(pca_ridge$x[, 1:2])
train_pca_ridge$target <- train_scaled$target

# Logistic model on PC1-PC2 (for boundary only)
ridge_pca_model <- glm(target ~ ., data = train_pca_ridge, family = "binomial")

# Grid for decision boundary
x_min <- min(train_pca_ridge$PC1) - 1
x_max <- max(train_pca_ridge$PC1) + 1
y_min <- min(train_pca_ridge$PC2) - 1
y_max <- max(train_pca_ridge$PC2) + 1

grid_df <- expand.grid(
  PC1 = seq(x_min, x_max, length.out = 200),
  PC2 = seq(y_min, y_max, length.out = 200)
)

grid_df$prob <- predict(ridge_pca_model, newdata = grid_df, type = "response")
grid_df$prediction <- factor(ifelse(grid_df$prob > 0.5, "tumor", "normal"),
                             levels = c("normal", "tumor"))

# ============================
# Get PCA explained variance
# ============================
expl_var_ridge <- round(summary(pca_ridge)$importance[2, 1:2] * 100, 1)
pc1_lab <- paste0("PC1 (", expl_var_ridge[1], "%)")
pc2_lab <- paste0("PC2 (", expl_var_ridge[2], "%)")

# ============================
# Plot – Train set decision boundary
# ============================
ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = train_pca_ridge, aes(x = PC1, y = PC2, color = target), size = 2.5, alpha = 0.9) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(title = "Ridge – Decision Boundary (Train Set)",
       subtitle = "Background = prediction, Point = true class",
       x = pc1_lab, y = pc2_lab) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")

# ============================
# PCA on Test Set
# ============================
pca_test_ridge <- predict(pca_ridge, newdata = test_scaled %>% select(all_of(selected_features)))
test_pca_ridge <- as.data.frame(pca_test_ridge[, 1:2])
test_pca_ridge$target <- test_scaled$target

# ============================
# Plot – Test set decision boundary
# ============================
ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = test_pca_ridge, aes(x = PC1, y = PC2, color = target), size = 2.5, alpha = 0.9) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(title = "Ridge – Decision Boundary (Test Set)",
       subtitle = "Background = prediction, Point = true class",
       x = pc1_lab, y = pc2_lab) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")





# ========================================================
# LASSO Logistic Regression – Using Lasso-Selected Features
# ========================================================

# 1. Prepare training and test matrices using Lasso-selected features
X_train_lasso <- as.matrix(train_scaled %>% select(all_of(selected_features)))
X_test_lasso  <- as.matrix(test_scaled  %>% select(all_of(selected_features)))

y_train <- train_scaled$target
y_test  <- test_scaled$target

# 2. Cross-validation to find optimal lambda (alpha = 1 for Lasso)
set.seed(42)
lasso_cv <- cv.glmnet(
  X_train_lasso, y_train,
  alpha = 1,
  family = "binomial",
  type.measure = "auc",
  nfolds = 10
)

# 3. Plot AUC vs log(lambda) with confidence ribbon
lambda_df_lasso <- data.frame(
  lambda = lasso_cv$lambda,
  AUC = lasso_cv$cvm,
  SD = lasso_cv$cvsd
)

ggplot(lambda_df_lasso, aes(x = log(lambda), y = AUC)) +
  geom_line(color = "#ff7f0e", linewidth = 1.2) +
  geom_ribbon(aes(ymin = AUC - SD, ymax = AUC + SD), alpha = 0.2, fill = "#ff7f0e") +
  geom_vline(xintercept = log(lasso_cv$lambda.min), linetype = "dashed", color = "#d62728") +
  annotate("text", x = log(lasso_cv$lambda.min), y = max(lambda_df_lasso$AUC),
           label = paste0("Best λ = ", round(lasso_cv$lambda.min, 4)),
           hjust = -0.1, color = "#d62728", size = 5) +
  theme_minimal(base_size = 14) +
  labs(title = "AUC vs log(Lambda) – Lasso (alpha = 1)",
       x = "log(Lambda)", y = "Cross-Validated AUC")

# 4. Train final Lasso model using best lambda
lasso_model <- glmnet(
  X_train_lasso, y_train,
  alpha = 1,
  lambda = lasso_cv$lambda.min,
  family = "binomial"
)

# 5. Predictions on TRAIN set
train_probs_lasso <- predict(lasso_model, X_train_lasso, type = "response")
train_preds_lasso <- factor(ifelse(train_probs_lasso > 0.5, "tumor", "normal"),
                            levels = c("normal", "tumor"))
conf_train_lasso <- confusionMatrix(train_preds_lasso, y_train)

# Compute F1-score for TRAIN set
cm_train <- conf_train_lasso$table
TP <- cm_train[2, 2]; FP <- cm_train[1, 2]
FN <- cm_train[2, 1]; TN <- cm_train[1, 1]
precision_train <- TP / (TP + FP)
recall_train <- TP / (TP + FN)
f1_train <- 2 * precision_train * recall_train / (precision_train + recall_train)

cat("Lasso – TRAIN performance:\n")
print(conf_train_lasso)
cat("\nTRAIN metrics:\n")
cat("   Precision:", round(precision_train, 3), "\n")
cat("   Recall:   ", round(recall_train, 3), "\n")
cat("   F1-score: ", round(f1_train, 3), "\n")

# 6. Predictions on TEST set
test_probs_lasso <- predict(lasso_model, X_test_lasso, type = "response")
test_preds_lasso <- factor(ifelse(test_probs_lasso > 0.5, "tumor", "normal"),
                           levels = c("normal", "tumor"))
conf_test_lasso <- confusionMatrix(test_preds_lasso, y_test)

# Compute F1-score for TEST set
cm_test <- conf_test_lasso$table
TP <- cm_test[2, 2]; FP <- cm_test[1, 2]
FN <- cm_test[2, 1]; TN <- cm_test[1, 1]
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * precision * recall / (precision + recall)

cat("\nLasso – TEST performance:\n")
print(conf_test_lasso)
cat("\nTEST metrics:\n")
cat("   Precision:", round(precision, 3), "\n")
cat("   Recall:   ", round(recall, 3), "\n")
cat("   F1-score: ", round(f1_score, 3), "\n")

# 7. ROC curve and AUC
roc_lasso_fs <- roc(response = as.numeric(y_test) == 2,
                 predictor = as.numeric(test_probs_lasso),
                 direction = "<", quiet = TRUE)
auc_lasso <- auc(roc_lasso_fs)

roc_lasso_fs <- data.frame(
  FPR = rev(1 - roc_lasso_fs$specificities),
  TPR = rev(roc_lasso_fs$sensitivities)
)
roc_lasso_fs <- rbind(c(0, 0), roc_lasso_fs, c(1, 1))
colnames(roc_lasso_fs) <- c("FPR", "TPR")

ggplot(roc_df_lasso, aes(x = FPR, y = TPR)) +
  geom_area(fill = "#ff7f0e", alpha = 0.2) +
  geom_line(color = "#ff7f0e", linewidth = 1.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  annotate("text", x = 0.7, y = 0.15,
           label = paste("AUC =", round(auc_lasso, 3)),
           size = 6, fontface = "italic", color = "black") +
  labs(title = "ROC Curve – Lasso (Feature Selection)",
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
  theme_minimal(base_size = 16)

# 8. Report final retained features
lasso_coef_final <- coef(lasso_model)
lasso_selected_final <- rownames(lasso_coef_final)[which(lasso_coef_final != 0)][-1]

cat("\nLasso retained", length(lasso_selected_final), "features (out of", length(selected_features), "provided):\n")
print(lasso_selected_final)

excluded_by_lasso <- setdiff(selected_features, lasso_selected_final)
cat("\nFeatures excluded by Lasso:\n")
print(excluded_by_lasso)

# 9. PCA visualization of TRAIN set
pca_lasso <- prcomp(train_scaled %>% select(all_of(selected_features)), center = TRUE, scale. = TRUE)
train_pca_lasso <- as.data.frame(pca_lasso$x[, 1:2])
train_pca_lasso$target <- train_scaled$target

# Fit logistic regression on PC1 and PC2 to generate decision boundary
lasso_pca_model <- glm(target ~ ., data = train_pca_lasso, family = "binomial")

# Create grid for plotting decision boundary
x_min <- min(train_pca_lasso$PC1) - 1
x_max <- max(train_pca_lasso$PC1) + 1
y_min <- min(train_pca_lasso$PC2) - 1
y_max <- max(train_pca_lasso$PC2) + 1

grid_df <- expand.grid(
  PC1 = seq(x_min, x_max, length.out = 200),
  PC2 = seq(y_min, y_max, length.out = 200)
)
grid_df$prob <- predict(lasso_pca_model, newdata = grid_df, type = "response")
grid_df$prediction <- factor(ifelse(grid_df$prob > 0.5, "tumor", "normal"),
                             levels = c("normal", "tumor"))

# Plot decision boundary on TRAIN set
# Calcola varianza spiegata dalle prime due componenti
explained_var_lasso <- round(summary(pca_lasso)$importance[2, 1:2] * 100, 1)
x_lab_lasso <- paste0("PC1 (", explained_var_lasso[1], "%)")
y_lab_lasso <- paste0("PC2 (", explained_var_lasso[2], "%)")

# Plot decision boundary on TRAIN set
ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = train_pca_lasso, aes(x = PC1, y = PC2, color = target), size = 2.5, alpha = 0.9) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(title = "Lasso – Decision Boundary (Train Set)",
       subtitle = "Background = prediction, Point = true class",
       x = x_lab_lasso, y = y_lab_lasso) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")

# Project TEST set on PCA components and plot
pca_test_lasso <- predict(pca_lasso, newdata = test_scaled %>% select(all_of(selected_features)))
test_pca_lasso <- as.data.frame(pca_test_lasso[, 1:2])
test_pca_lasso$target <- test_scaled$target

ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = test_pca_lasso, aes(x = PC1, y = PC2, color = target), size = 2.5, alpha = 0.9) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(title = "Lasso – Decision Boundary (Test Set)",
       subtitle = "Background = prediction, Point = true class",
       x = x_lab_lasso, y = y_lab_lasso) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")






# ================================================================
# Naive Bayes Classifier – Full Feature Set vs Lasso-Selected Features
# ================================================================
# This section applies the Naive Bayes classifier to a binary classification 
# task (normal vs tumor). First, the model is trained on all available predictors. 
# Then, a second version will be trained using only the features selected by Lasso.
# Performance will be evaluated using F1-score, ROC curves, and decision boundaries.
# ================================================================


# ====================================
# 1. Training on all predictors
# ====================================

train_nb <- train_scaled
test_nb  <- test_scaled

nb_model <- naiveBayes(target ~ ., data = train_nb)

# ====================================
# 2. Predictions on TRAIN set
# ====================================

train_probs_nb <- predict(nb_model, newdata = train_nb, type = "raw")[, "tumor"]
train_preds_nb <- factor(ifelse(train_probs_nb > 0.5, "tumor", "normal"),
                         levels = c("normal", "tumor"))
conf_nb_train <- confusionMatrix(train_preds_nb, train_nb$target)

# Compute metrics – TRAIN
cm_train <- conf_nb_train$table
TP <- cm_train[2, 2]; FP <- cm_train[1, 2]
FN <- cm_train[2, 1]; TN <- cm_train[1, 1]
precision_train <- TP / (TP + FP)
recall_train    <- TP / (TP + FN)
f1_train        <- 2 * precision_train * recall_train / (precision_train + recall_train)

cat("Confusion Matrix – TRAIN (Naive Bayes):\n")
print(conf_nb_train)
cat("\nTRAIN Metrics:\n")
cat("   Precision:", round(precision_train, 3), "\n")
cat("   Recall:   ", round(recall_train, 3), "\n")
cat("   F1-score: ", round(f1_train, 3), "\n")

# ====================================
# 3. Predictions on TEST set
# ====================================

test_probs_nb <- predict(nb_model, newdata = test_nb, type = "raw")[, "tumor"]
test_preds_nb <- factor(ifelse(test_probs_nb > 0.5, "tumor", "normal"),
                        levels = c("normal", "tumor"))
conf_nb <- confusionMatrix(test_preds_nb, test_nb$target)

# Compute metrics – TEST
cm <- conf_nb$table
TP <- cm[2, 2]; FP <- cm[1, 2]
FN <- cm[2, 1]; TN <- cm[1, 1]
precision <- TP / (TP + FP)
recall    <- TP / (TP + FN)
f1_score  <- 2 * precision * recall / (precision + recall)

cat("\nConfusion Matrix – TEST (Naive Bayes):\n")
print(conf_nb)
cat("\nTEST Metrics:\n")
cat("   Precision:", round(precision, 3), "\n")
cat("   Recall:   ", round(recall, 3), "\n")
cat("   F1-score: ", round(f1_score, 3), "\n")

# ====================================
# 4. ROC Curve and AUC
# ====================================

roc_nb <- roc(response = as.numeric(test_nb$target) == 2,
              predictor = test_probs_nb,
              direction = "<", quiet = TRUE)
auc_nb <- auc(roc_nb)

roc_df_nb <- data.frame(
  FPR = rev(1 - roc_nb$specificities),
  TPR = rev(roc_nb$sensitivities)
)
roc_df_nb <- rbind(c(0, 0), roc_df_nb, c(1, 1))
colnames(roc_df_nb) <- c("FPR", "TPR")

ggplot(roc_df_nb, aes(x = FPR, y = TPR)) +
  geom_area(fill = "#9467bd", alpha = 0.2) +
  geom_line(color = "#9467bd", linewidth = 1.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  annotate("text", x = 0.7, y = 0.15,
           label = paste("AUC =", round(auc_nb, 3)),
           size = 6, fontface = "italic", color = "black") +
  labs(title = "ROC Curve – Naive Bayes (All Predictors)",
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
  theme_minimal(base_size = 16)

# ====================================
# 5. PCA projection (Train set)
# ====================================

pca_nb <- prcomp(train_scaled %>% select(-target), center = TRUE, scale. = TRUE)
train_pca <- as.data.frame(pca_nb$x[, 1:2])
train_pca$target <- train_scaled$target

# ====================================
# 6. Train Naive Bayes on PC1 + PC2
# ====================================

nb_model_2d <- naiveBayes(target ~ ., data = train_pca)

# ====================================
# 7. Grid for decision boundary
# ====================================

x_min <- min(train_pca$PC1) - 1
x_max <- max(train_pca$PC1) + 1
y_min <- min(train_pca$PC2) - 1
y_max <- max(train_pca$PC2) + 1

grid_df <- expand.grid(
  PC1 = seq(x_min, x_max, length.out = 200),
  PC2 = seq(y_min, y_max, length.out = 200)
)
grid_df$prediction <- predict(nb_model_2d, newdata = grid_df)

# ====================================
# 8. Plot – Decision Boundary (Train)
# ====================================

ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = train_pca, aes(x = PC1, y = PC2, color = target), size = 2.5, alpha = 0.9) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(title = "Naive Bayes – Decision Boundary (Train Set)",
       subtitle = "Background = prediction, Point = true class",
       x = "PC1", y = "PC2") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")

# ====================================
# 9. Project TEST set onto PCA
# ====================================

pca_test <- predict(pca_nb, newdata = test_scaled %>% select(-target))
test_pca <- as.data.frame(pca_test[, 1:2])
test_pca$target <- test_scaled$target

# ====================================
# 10. Plot – Decision Boundary (Test)
# ====================================

ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = test_pca, aes(x = PC1, y = PC2, color = target, shape = target), size = 2.8, alpha = 0.9) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  scale_shape_manual(values = c("normal" = 16, "tumor" = 17)) +
  labs(title = "Naive Bayes – Decision Boundary (Test Set)",
       subtitle = "Background = prediction, Point = true class",
       x = "PC1", y = "PC2") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")








# ================================================================
# Naive Bayes Classifier – Using Lasso-Selected Features
# ================================================================
# In this section, we train a Naive Bayes model using only the subset 
# of predictors selected via Lasso regularization. This approach allows 
# us to reduce dimensionality and potentially improve generalization. 
# The performance is evaluated on the test set through confusion matrix, 
# F1-score, and ROC curve.
# ================================================================



# 1. Prepare training set with selected features
X_lasso <- train_scaled %>% select(all_of(selected_features))
train_lasso_df <- as.data.frame(X_lasso)
train_lasso_df$target <- train_scaled$target

# 2. Train Naive Bayes model
nb_model_lasso <- naiveBayes(target ~ ., data = train_lasso_df)

# 3. Prepare test set using same features
X_test_lasso <- test_scaled %>% select(all_of(selected_features))
test_lasso_df <- as.data.frame(X_test_lasso)
test_lasso_df$target <- test_scaled$target

# 4. Predict on test set
test_probs_nb_lasso <- predict(nb_model_lasso, newdata = test_lasso_df, type = "raw")[, "tumor"]
test_preds_nb_lasso <- factor(ifelse(test_probs_nb_lasso > 0.5, "tumor", "normal"),
                              levels = c("normal", "tumor"))

# 5. Confusion matrix
conf_nb_lasso <- confusionMatrix(test_preds_nb_lasso, test_lasso_df$target)
cat("Confusion Matrix – TEST (Naive Bayes, Lasso Features):\n")
print(conf_nb_lasso)

# 6. Compute AUC and ROC
roc_nb_fs <- roc(response = as.numeric(test_lasso_df$target) == 2,
                    predictor = test_probs_nb_lasso,
                    direction = "<", quiet = TRUE)
auc_nb_lasso <- auc(roc_nb_lasso)

roc_nb_fs <- data.frame(
  FPR = rev(1 - roc_nb_fs$specificities),
  TPR = rev(roc_nb_fs$sensitivities)
)
roc_nb_fs <- rbind(c(0, 0), roc_nb_fs, c(1, 1))
colnames(roc_nb_fs) <- c("FPR", "TPR")

# 7. Plot ROC Curve
ggplot(roc_nb_fs, aes(x = FPR, y = TPR)) +
  geom_area(fill = "#bcbd22", alpha = 0.2) +
  geom_line(color = "#bcbd22", linewidth = 1.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  annotate("text", x = 0.7, y = 0.15,
           label = paste("AUC =", round(auc_nb_lasso, 3)),
           size = 6, fontface = "italic", color = "black") +
  labs(title = "ROC Curve – Naive Bayes (Lasso Features)",
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
  theme_minimal(base_size = 16)

# ===========================
# PCA visualization
# ===========================

pca_nb_lasso <- prcomp(train_lasso_df %>% select(-target), center = TRUE, scale. = TRUE)
train_pca_nb <- as.data.frame(pca_nb_lasso$x[, 1:2])
train_pca_nb$target <- train_lasso_df$target

# Supponendo che 'pca_nb_lasso' sia il risultato di prcomp sul training set con le feature selezionate
# Proietta il test set sulle stesse componenti principali
test_pca_nb <- predict(pca_nb_lasso, newdata = test_lasso_df %>% select(-target))
test_pca_nb <- as.data.frame(test_pca_nb[, 1:2])
test_pca_nb$target <- test_lasso_df$target


# Fit Naive Bayes on PC1 and PC2 (for visualization)
nb_model_2d_lasso <- naiveBayes(target ~ ., data = train_pca_nb)

# Create grid for decision boundary
x_min <- min(train_pca_nb$PC1) - 1
x_max <- max(train_pca_nb$PC1) + 1
y_min <- min(train_pca_nb$PC2) - 1
y_max <- max(train_pca_nb$PC2) + 1

grid_df <- expand.grid(
  PC1 = seq(x_min, x_max, length.out = 200),
  PC2 = seq(y_min, y_max, length.out = 200)
)
grid_df$prediction <- predict(nb_model_2d_lasso, newdata = grid_df)

# Explained variance labels
explained_var_nb_lasso <- round(summary(pca_nb_lasso)$importance[2, 1:2] * 100, 1)
x_lab_nb <- paste0("PC1 (", explained_var_nb_lasso[1], "%)")
y_lab_nb <- paste0("PC2 (", explained_var_nb_lasso[2], "%)")

# Plot decision boundary on TRAIN set with explained variance
ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = train_pca_nb, aes(x = PC1, y = PC2, color = target), size = 2.5, alpha = 0.9) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(title = "Naive Bayes – Decision Boundary (Train Set)",
       subtitle = "Background = prediction, Point = true class",
       x = x_lab_nb, y = y_lab_nb) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")



# Plot – Test set decision boundary
ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = test_pca_nb, aes(x = PC1, y = PC2, color = target), size = 2.5, alpha = 0.9) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  labs(title = "Naive Bayes – Decision Boundary (Test Set)",
       subtitle = "Background = prediction, Point = true class",
       x = x_lab_nb, y = y_lab_nb) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")










# ================================================================
# K-Nearest Neighbors Classifier – Using All Predictors
# ================================================================
# The K-Nearest Neighbors (KNN) classifier is a non-parametric method 
# that classifies samples based on the majority vote of their 'k' nearest neighbors. 
# Its simplicity makes it effective in low-dimensional spaces, but it can suffer 
# from the curse of dimensionality. We apply KNN to our scaled dataset:
# (1) using all predictors without prior feature selection,
# (2) tuning the number of neighbors (k) via cross-validation to maximize AUC.
# The performance is then evaluated on both training and test sets.
# ================================================================

# ====================================
# 1. Prepare data
# ====================================
X_train <- train_scaled %>% select(-target)
y_train <- train_scaled$target

X_test  <- test_scaled %>% select(-target)
y_test  <- test_scaled$target

# ====================================
# 2. Cross-validation to select optimal K
# ====================================
set.seed(42)
cv_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

knn_grid <- expand.grid(k = seq(1, 25, by = 2))

knn_model <- train(
  x = X_train,
  y = y_train,
  method = "knn",
  trControl = cv_control,
  tuneGrid = knn_grid,
  metric = "ROC",
  preProcess = NULL
)

best_k <- knn_model$bestTune$k
cat("Best K selected via cross-validation:", best_k, "\n")

# ====================================
# 3. Prediction on train set
# ====================================
train_probs_knn <- predict(knn_model, newdata = X_train, type = "prob")[, "tumor"]
train_preds_knn <- predict(knn_model, newdata = X_train)

conf_knn_train <- confusionMatrix(train_preds_knn, y_train)
print(conf_knn_train)

cm_train <- conf_knn_train$table
TP_train <- cm_train[2, 2]; FP_train <- cm_train[1, 2]
FN_train <- cm_train[2, 1]; TN_train <- cm_train[1, 1]

precision_train <- TP_train / (TP_train + FP_train)
recall_train    <- TP_train / (TP_train + FN_train)
f1_train        <- 2 * precision_train * recall_train / (precision_train + recall_train)

cat("\nKNN (k =", best_k, ") – Train Set Metrics:\n")
cat("   Precision:", round(precision_train, 3), "\n")
cat("   Recall:   ", round(recall_train, 3), "\n")
cat("   F1-score: ", round(f1_train, 3), "\n")

# ====================================
# 4. Prediction on test set
# ====================================
test_probs_knn <- predict(knn_model, newdata = X_test, type = "prob")[, "tumor"]
test_preds_knn <- predict(knn_model, newdata = X_test)

conf_knn <- confusionMatrix(test_preds_knn, y_test)
print(conf_knn)

cm <- conf_knn$table
TP <- cm[2, 2]; FP <- cm[1, 2]
FN <- cm[2, 1]; TN <- cm[1, 1]

precision <- TP / (TP + FP)
recall    <- TP / (TP + FN)
f1        <- 2 * precision * recall / (precision + recall)

cat("\nKNN (k =", best_k, ") – Test Set Metrics:\n")
cat("   Precision:", round(precision, 3), "\n")
cat("   Recall:   ", round(recall, 3), "\n")
cat("   F1-score: ", round(f1, 3), "\n")

# ====================================
# 5. ROC and AUC (test set)
# ====================================
roc_knn <- roc(response = as.numeric(y_test) == 2,
               predictor = test_probs_knn,
               direction = "<", quiet = TRUE)
auc_knn <- auc(roc_knn)

roc_df_knn <- data.frame(
  FPR = rev(1 - roc_knn$specificities),
  TPR = rev(roc_knn$sensitivities)
)
roc_df_knn <- rbind(c(0, 0), roc_df_knn, c(1, 1))
colnames(roc_df_knn) <- c("FPR", "TPR")

ggplot(roc_df_knn, aes(x = FPR, y = TPR)) +
  geom_area(fill = "#1f77b4", alpha = 0.2) +
  geom_line(color = "#1f77b4", linewidth = 1.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  annotate("text", x = 0.7, y = 0.15,
           label = paste("AUC =", round(auc_knn, 3)),
           size = 6, fontface = "italic", color = "black") +
  labs(title = paste("ROC Curve – KNN (k =", best_k, ")"),
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
  theme_minimal(base_size = 16)

# ====================================
# 6. PCA Projection and Decision Visualization (Train set)
# ====================================
pca_train <- prcomp(X_train, center = TRUE, scale. = TRUE)
train_pca <- as.data.frame(pca_train$x[, 1:2])
train_pca$target <- y_train

knn_model_2d <- knn3(train_pca[, 1:2], y_train, k = best_k)

# Grid for decision boundary
x_min <- min(train_pca$PC1) - 1
x_max <- max(train_pca$PC1) + 1
y_min <- min(train_pca$PC2) - 1
y_max <- max(train_pca$PC2) + 1

grid_df <- expand.grid(
  PC1 = seq(x_min, x_max, length.out = 200),
  PC2 = seq(y_min, y_max, length.out = 200)
)

grid_preds <- predict(knn_model_2d, newdata = grid_df, type = "class")
grid_df$prediction <- factor(grid_preds, levels = levels(y_train))

explained_var_train <- round(summary(pca_train)$importance[2, 1:2] * 100, 1)
pc1_lab <- paste0("PC1 (", explained_var_train[1], "%)")
pc2_lab <- paste0("PC2 (", explained_var_train[2], "%)")

# Plot – Decision Boundary (Train Set) con triangoli per entrambe le classi
ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = train_pca, aes(x = PC1, y = PC2, color = target, shape = target), size = 2.5, alpha = 0.9) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  scale_shape_manual(values = c("normal" = 16, "tumor" = 17)) +  # triangolo per entrambe le classi
  labs(title = paste("KNN (k =", best_k, ") – Decision Boundary (Train Set)"),
       subtitle = "Background = prediction, Point = true class",
       x = pc1_lab, y = pc2_lab) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")

# ====================================
# 7. PCA Projection and Decision Visualization (Test set)
# ====================================
pca_test <- predict(pca_train, newdata = X_test)
test_pca <- as.data.frame(pca_test[, 1:2])
test_pca$target <- y_test

ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = test_pca, aes(x = PC1, y = PC2, color = target, shape = target), size = 2.8, alpha = 0.9) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  scale_shape_manual(values = c("normal" = 16, "tumor" = 17)) +  # triangolo per entrambe le classi
  labs(title = paste("KNN (k =", best_k, ") – PCA Projection on Test Set"),
       subtitle = "Color = predicted class, Shape = true class",
       x = pc1_lab, y = pc2_lab) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")





# ================================================================
# K-Nearest Neighbors Classifier – Using Lasso-Selected Features
# ================================================================
# This implementation evaluates KNN performance after applying  
# Lasso-based feature selection. Feature selection helps mitigate  
# the curse of dimensionality by retaining only the most informative  
# variables, which is particularly useful for distance-based classifiers  
# like KNN. The best 'k' is selected via cross-validation based on AUC.
# ================================================================

# ====================================
# 1. Prepare data with selected features
# ====================================
X_train_fs <- train_scaled %>% select(all_of(selected_features))
X_test_fs  <- test_scaled %>% select(all_of(selected_features))

y_train <- train_scaled$target
y_test  <- test_scaled$target

# ====================================
# 2. Cross-validation to find optimal K
# ====================================
set.seed(42)
cv_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

knn_grid <- expand.grid(k = seq(1, 25, by = 2))

knn_model_fs <- train(
  x = X_train_fs,
  y = y_train,
  method = "knn",
  trControl = cv_control,
  tuneGrid = knn_grid,
  metric = "ROC"
)

best_k_fs <- knn_model_fs$bestTune$k
cat("Best K selected (Lasso features):", best_k_fs, "\n")

# ====================================
# 3. Prediction on train set
# ====================================
train_probs_knn_fs <- predict(knn_model_fs, newdata = X_train_fs, type = "prob")[, "tumor"]
train_preds_knn_fs <- predict(knn_model_fs, newdata = X_train_fs)

conf_knn_train <- confusionMatrix(train_preds_knn_fs, y_train)
print(conf_knn_train)

cm_train <- conf_knn_train$table
TP_train <- cm_train[2, 2]; FP_train <- cm_train[1, 2]
FN_train <- cm_train[2, 1]; TN_train <- cm_train[1, 1]

precision_train <- TP_train / (TP_train + FP_train)
recall_train    <- TP_train / (TP_train + FN_train)
f1_train        <- 2 * precision_train * recall_train / (precision_train + recall_train)

cat("\nKNN post-Lasso (k =", best_k_fs, ") – Train Set Metrics:\n")
cat("   Precision:", round(precision_train, 3), "\n")
cat("   Recall:   ", round(recall_train, 3), "\n")
cat("   F1-score: ", round(f1_train, 3), "\n")

# ====================================
# 4. Prediction on test set
# ====================================
test_probs_knn_fs <- predict(knn_model_fs, newdata = X_test_fs, type = "prob")[, "tumor"]
test_preds_knn_fs <- predict(knn_model_fs, newdata = X_test_fs)

conf_knn_fs <- confusionMatrix(test_preds_knn_fs, y_test)
print(conf_knn_fs)

cm <- conf_knn_fs$table
TP <- cm[2, 2]; FP <- cm[1, 2]
FN <- cm[2, 1]; TN <- cm[1, 1]

precision <- TP / (TP + FP)
recall    <- TP / (TP + FN)
f1        <- 2 * precision * recall / (precision + recall)

cat("\nKNN post-Lasso (k =", best_k_fs, ") – Test Set Metrics:\n")
cat("   Precision:", round(precision, 3), "\n")
cat("   Recall:   ", round(recall, 3), "\n")
cat("   F1-score: ", round(f1, 3), "\n")

# ====================================
# 5. ROC Curve and AUC (Test set)
# ====================================
roc_knn_fs <- roc(response = as.numeric(y_test) == 2,
                  predictor = test_probs_knn_fs,
                  direction = "<", quiet = TRUE)
auc_knn_fs <- auc(roc_knn_fs)

roc_knn_fs <- data.frame(
  FPR = rev(1 - roc_knn_fs$specificities),
  TPR = rev(roc_knn_fs$sensitivities)
)
roc_knn_fs <- rbind(c(0, 0), roc_knn_fs, c(1, 1))
colnames(roc_knn_fs) <- c("FPR", "TPR")

ggplot(roc_knn_fs, aes(x = FPR, y = TPR)) +
  geom_area(fill = "#1f77b4", alpha = 0.2) +
  geom_line(color = "#1f77b4", linewidth = 1.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  annotate("text", x = 0.7, y = 0.15,
           label = paste("AUC =", round(auc_knn_fs, 3)),
           size = 6, fontface = "italic", color = "black") +
  labs(title = paste("ROC Curve – KNN post-Lasso (k =", best_k_fs, ")"),
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
  theme_minimal(base_size = 16)

# ====================================
# 6. PCA and decision boundary visualization (Train set)
# ====================================
pca_train <- prcomp(X_train_fs, center = TRUE, scale. = TRUE)
train_pca <- as.data.frame(pca_train$x[, 1:2])
train_pca$target <- y_train

knn_model_2d <- knn3(train_pca[, 1:2], y_train, k = best_k_fs)

# Grid for decision boundary
x_min <- min(train_pca$PC1) - 1
x_max <- max(train_pca$PC1) + 1
y_min <- min(train_pca$PC2) - 1
y_max <- max(train_pca$PC2) + 1

grid_df <- expand.grid(
  PC1 = seq(x_min, x_max, length.out = 200),
  PC2 = seq(y_min, y_max, length.out = 200)
)

grid_preds <- predict(knn_model_2d, newdata = grid_df, type = "class")
grid_df$prediction <- factor(grid_preds, levels = levels(y_train))

explained_var_train <- round(summary(pca_train)$importance[2, 1:2] * 100, 1)
pc1_lab <- paste0("PC1 (", explained_var_train[1], "%)")
pc2_lab <- paste0("PC2 (", explained_var_train[2], "%)")

ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = train_pca, aes(x = PC1, y = PC2, color = target, shape = target), size = 2.5, alpha = 0.9) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  scale_shape_manual(values = c("normal" = 16, "tumor" = 17)) + # triangolo per entrambe le classi
  labs(title = paste("KNN post-Lasso (k =", best_k_fs, ") – Decision Boundary (Train Set)"),
       subtitle = "Background = prediction, Point = true class",
       x = pc1_lab, y = pc2_lab) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")

# ====================================
# 7. PCA Projection and decision boundary (Test set)
# ====================================
pca_test <- predict(pca_train, newdata = X_test_fs)
test_pca <- as.data.frame(pca_test[, 1:2])
test_pca$target <- y_test

ggplot() +
  geom_tile(data = grid_df, aes(x = PC1, y = PC2, fill = prediction), alpha = 0.25) +
  scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  geom_point(data = test_pca, aes(x = PC1, y = PC2, color = target, shape = target), size = 2.8, alpha = 0.9) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  scale_shape_manual(values = c("normal" = 16, "tumor" = 17)) + # triangolo per entrambe le classi
  labs(title = paste("KNN post-Lasso (k =", best_k_fs, ") – PCA Projection on Test Set"),
       subtitle = "Color = predicted class, Shape = true class",
       x = pc1_lab, y = pc2_lab) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")





# ================================================================
# Comparative ROC Curves – All Models (Full Features & Post-Lasso)
# ================================================================
# This section builds comparative ROC plots for all classifiers
# using both the full feature set and the Lasso-selected subset.
# It enables direct visual comparison of model performance in terms 
# of trade-offs between sensitivity and specificity on the test set.
# ================================================================


# ===================================================
# Comparative ROC – Using All Original Features
# ===================================================

# 1. Merge individual ROC data into a single dataframe
roc_all <- bind_rows(
  roc_df_en    %>% mutate(model = "ElasticNet"),
  roc_ridge_df %>% mutate(model = "Ridge"),
  roc_df_lasso %>% mutate(model = "Lasso"),
  roc_df_nb    %>% mutate(model = "Naive Bayes"),
  roc_df_knn   %>% mutate(model = "KNN")
)

# 2. Define consistent color palette
model_colors <- c(
  "ElasticNet"   = "#2ca02c",  # green
  "Ridge"        = "#1f77b4",  # blue
  "Lasso"        = "#ff7f0e",  # orange
  "Naive Bayes"  = "#9467bd",  # purple
  "KNN"          = "#d62728"   # red
)

# 3. Plot ROC curves
ggplot(roc_all, aes(x = FPR, y = TPR, color = model)) +
  geom_line(linewidth = 1.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey70", linewidth = 1) +
  scale_color_manual(values = model_colors) +
  coord_fixed() +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 18, face = "bold"),
    plot.subtitle = element_text(size = 14, face = "italic"),
    axis.title = element_text(size = 13),
    axis.text = element_text(size = 11),
    legend.title = element_text(size = 13),
    legend.text = element_text(size = 12)
  ) +
  labs(
    title = "Comparative ROC Curve – Test Set (All Features)",
    subtitle = "ElasticNet, Ridge, Lasso, Naive Bayes, and KNN",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    color = "Model"
  )


# ===================================================
# Comparative ROC – Using Lasso-Selected Features
# ===================================================

# 1. Merge individual ROC data from feature-selected models
roc_all_fs <- bind_rows(
  roc_en_fs    %>% mutate(model = "ElasticNet"),
  roc_ridge_fs %>% mutate(model = "Ridge"),
  roc_lasso_fs %>% mutate(model = "Lasso"),
  roc_nb_fs    %>% mutate(model = "Naive Bayes"),
  roc_knn_fs   %>% mutate(model = "KNN")
)

# 2. (Reuse) Color palette for consistency
model_colors <- c(
  "ElasticNet"   = "#2ca02c",
  "Ridge"        = "#1f77b4",
  "Lasso"        = "#ff7f0e",
  "Naive Bayes"  = "#9467bd",
  "KNN"          = "#d62728"
)

# 3. Plot ROC curves
ggplot(roc_all_fs, aes(x = FPR, y = TPR, color = model)) +
  geom_line(linewidth = 1.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey70", linewidth = 1) +
  scale_color_manual(values = model_colors) +
  coord_fixed() +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 18, face = "bold"),
    plot.subtitle = element_text(size = 14, face = "italic"),
    axis.title = element_text(size = 13),
    axis.text = element_text(size = 11),
    legend.title = element_text(size = 13),
    legend.text = element_text(size = 12)
  ) +
  labs(
    title = "Comparative ROC Curve – Test Set (Post-Lasso)",
    subtitle = "ElasticNet, Ridge, Lasso, Naive Bayes, and KNN",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    color = "Model"
  )







