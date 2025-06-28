# ===============================================================
# STEP – Load Saved Processed Datasets
# ===============================================================

# Set the working directory to the location where the processed files are saved
output_dir <- "Processed"
setwd(output_dir)  # Make sure the path is correct

# ---------------------------------------
# 1. Load the predictor datasets
# ---------------------------------------
X_raw_scaled <- read.csv("X_raw_scaled.csv")
X_clean_scaled <- read.csv("X_clean_scaled.csv")
X_pca <- read.csv("X_pca.csv")

# Check if the datasets were loaded correctly
cat("Loaded X_raw_scaled, X_clean_scaled, and X_pca\n")
str(X_raw_scaled)  
str(X_clean_scaled)
str(X_pca)

# ---------------------------------------
# 2. Remove non-predictive columns from X_clean_scaled
# ---------------------------------------

cols_to_remove <- c("subject.", "is_outlier", "Mahalanobis_Dist")
cols_existing <- intersect(cols_to_remove, colnames(X_clean_scaled))

if (length(cols_existing) > 0) {
  X_clean_scaled <- X_clean_scaled[, !(colnames(X_clean_scaled) %in% cols_existing)]
  cat("🔍 Removed non-predictive columns:", paste(cols_existing, collapse = ", "), "\n")
} else {
  cat("✅ No non-predictive columns to remove.\n")
}

# ---------------------------------------
# 3. Load the target datasets
# ---------------------------------------
y_total_raw <- read.csv("y_total_raw.csv")$y_total_raw
y_motor_raw <- read.csv("y_motor_raw.csv")$y_motor_raw

y_total_clean <- read.csv("y_total_clean.csv")$y_total_clean
y_motor_clean <- read.csv("y_motor_clean.csv")$y_motor_clean

y_total_pca <- read.csv("y_total_pca.csv")$y_total_pca
y_motor_pca <- read.csv("y_motor_pca.csv")$y_motor_pca

cat("Loaded y_total_raw, y_motor_raw, y_total_clean, y_motor_clean, y_total_pca, y_motor_pca\n")
str(y_total_raw)
str(y_motor_raw)
str(y_total_clean)
str(y_motor_clean)

# ------------------------------
# 4. Prepare training and test sets
# ------------------------------
set.seed(42)
train_index <- sample(seq_len(nrow(X_clean_scaled)), size = 0.8 * nrow(X_clean_scaled))

X_train <- X_clean_scaled[train_index, ]
X_test  <- X_clean_scaled[-train_index, ]

y_train_total <- y_total_clean[train_index]
y_test_total  <- y_total_clean[-train_index]

y_train_motor <- y_motor_clean[train_index]
y_test_motor  <- y_motor_clean[-train_index]

cat("X_train dimensions:", dim(X_train), "\n")
cat("X_test dimensions:", dim(X_test), "\n")
cat("y_train_total length:", length(y_train_total), "\n")
cat("y_test_total length:", length(y_test_total), "\n")
cat("y_train_motor length:", length(y_train_motor), "\n")
cat("y_test_motor length:", length(y_test_motor), "\n")
# ===============================================================
# STEP 5.1 – LASSO Regression for total_UPDRS
# ===============================================================

library(glmnet)
library(caret)
library(Metrics)

cat("\n📌 Training LASSO model for target: total_UPDRS\n")

# ================================
# 1. Data Preparation
# ================================

# Exclude non-predictive columns and the target variable
X_train_matrix <- as.matrix(X_train[, !colnames(X_train) %in% c("is_outlier", "total_UPDRS", "motor_UPDRS")])
X_test_matrix  <- as.matrix(X_test[, !colnames(X_test) %in% c("is_outlier", "total_UPDRS", "motor_UPDRS")])

# Targets
y_train_total  <- y_train_total
y_test_total   <- y_test_total

# ================================
# 2. Cross-validation with glmnet
# ================================
set.seed(123)

lasso_total_cv <- cv.glmnet(
  x = X_train_matrix,
  y = y_train_total,
  alpha = 1,                 # LASSO penalty
  nfolds = 10,
  standardize = FALSE        # data already standardized
)

# ================================
# 3. Cross-Validation Curve Plot
# ================================
library(ggplot2)

best_lambda <- lasso_total_cv$lambda.min

cv_df <- data.frame(
  lambda = lasso_total_cv$lambda,
  log_lambda = log(lasso_total_cv$lambda),
  mse = lasso_total_cv$cvm,
  mse_sd = lasso_total_cv$cvsd
)

ggplot(cv_df, aes(x = log_lambda, y = mse)) +
  geom_point(color = "lightgreen", size = 2) +
  geom_errorbar(aes(ymin = mse - mse_sd, ymax = mse + mse_sd), width = 0.1, color = "gray") +
  geom_vline(xintercept = log(best_lambda), color = "red", linetype = "dashed") +
  labs(
    title = "LASSO – Cross-Validated MSE vs Log(Lambda)",
    subtitle = paste("Optimal lambda =", signif(best_lambda, 4)),
    x = "log(Lambda)",
    y = "Mean Squared Error (MSE)"
  ) +
  theme_minimal()

# ================================
# 4. Train Final Model with Optimal Lambda
# ================================
cat("\n✅ Optimal lambda:", best_lambda, "\n")

lasso_total_final <- glmnet(
  x = X_train_matrix,
  y = y_train_total,
  alpha = 1,
  lambda = best_lambda,
  standardize = FALSE
)

# ================================
# 5. Prediction on Test Set
# ================================
pred_total_lasso <- predict(lasso_total_final, s = best_lambda, newx = X_test_matrix)

# ================================
# 6. Model Evaluation
# ================================
mse_lasso_total <- mse(y_test_total, pred_total_lasso)
ss_res <- sum((y_test_total - as.numeric(pred_total_lasso))^2)
ss_tot <- sum((y_test_total - mean(y_test_total))^2)
r2_lasso_total <- 1 - ss_res / ss_tot

cat("\n📊 LASSO Performance on total_UPDRS:\n")
cat("   ➤ MSE =", round(mse_lasso_total, 4), "\n")
cat("   ➤ R²  =", round(r2_lasso_total, 4), "\n")

# ================================
# 7. Selected Coefficients
# ================================
coef_lasso_total <- coef(lasso_total_final)
selected_features <- rownames(coef_lasso_total)[which(coef_lasso_total != 0)]

cat("\n📌 Selected variables by LASSO model:\n")
print(selected_features)

# ================================
# 8. Print LASSO Model Formula
# ================================

# Extract coefficients into a dataframe
coef_df <- as.data.frame(as.matrix(coef_lasso_total))
colnames(coef_df) <- "Coefficient"
coef_df$Variable <- rownames(coef_df)

# Remove zero coefficients
coef_df <- coef_df[coef_df$Coefficient != 0, ]

# Optional: sort by absolute coefficient value
coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]

# Build model formula as a string
intercept <- coef_df$Coefficient[coef_df$Variable == "(Intercept)"]
terms <- coef_df[coef_df$Variable != "(Intercept)", ]
formula_string <- paste0("total_UPDRS = ", round(intercept, 4))

for (i in 1:nrow(terms)) {
  coef_val <- round(terms$Coefficient[i], 4)
  var_name <- terms$Variable[i]
  sign <- ifelse(coef_val >= 0, " + ", " - ")
  formula_string <- paste0(formula_string, sign, abs(coef_val), "*", var_name)
}

cat("\n🧮 Final LASSO Model Formula:\n")
cat(formula_string, "\n")


# ===============================================================
# STEP 5.2 – Ridge Regression for total_UPDRS
# ===============================================================

library(glmnet)
library(Metrics)
library(ggplot2)

cat("\n📌 Training Ridge model for target: total_UPDRS\n")

# ================================
# 1. Data Preparation
# ================================

# Exclude non-predictive columns and the target variable
X_train_matrix <- as.matrix(X_train[, !colnames(X_train) %in% c("is_outlier", "total_UPDRS", "motor_UPDRS")])
X_test_matrix  <- as.matrix(X_test[, !colnames(X_test) %in% c("is_outlier", "total_UPDRS", "motor_UPDRS")])

# Display matrix dimensions
cat("\nDimension of X_train_matrix:", dim(X_train_matrix), "\n")
cat("Dimension of X_test_matrix:", dim(X_test_matrix), "\n")

# ================================
# 2. Ridge Cross-validation
# ================================
set.seed(123)

ridge_total_cv <- cv.glmnet(
  x = X_train_matrix,
  y = y_train_total,
  alpha = 0,                # Ridge penalty
  nfolds = 10,
  standardize = FALSE       # data already standardized
)

# ================================
# 3. Cross-Validation Curve Plot
# ================================
best_lambda_ridge <- ridge_total_cv$lambda.min

ridge_df <- data.frame(
  lambda = ridge_total_cv$lambda,
  log_lambda = log(ridge_total_cv$lambda),
  mse = ridge_total_cv$cvm,
  mse_sd = ridge_total_cv$cvsd
)

ggplot(ridge_df, aes(x = log_lambda, y = mse)) +
  geom_point(color = "lightgreen", size = 2) +
  geom_errorbar(aes(ymin = mse - mse_sd, ymax = mse + mse_sd), width = 0.1, color = "gray") +
  geom_vline(xintercept = log(best_lambda_ridge), linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title = "Ridge – Cross-Validated MSE vs Log(Lambda)",
    subtitle = paste("Optimal lambda =", signif(best_lambda_ridge, 4)),
    x = "log(Lambda)",
    y = "Mean Squared Error (MSE)"
  ) +
  theme_minimal()

# ================================
# 4. Train Final Model with Optimal Lambda
# ================================
ridge_total_final <- glmnet(
  x = X_train_matrix,
  y = y_train_total,
  alpha = 0,
  lambda = best_lambda_ridge,
  standardize = FALSE
)

# ================================
# 5. Prediction & Evaluation
# ================================
pred_total_ridge <- predict(ridge_total_final, s = best_lambda_ridge, newx = X_test_matrix)
pred_total_ridge <- as.numeric(pred_total_ridge)

# Compute metrics
mse_ridge <- mse(y_test_total, pred_total_ridge)
r2_ridge <- 1 - sum((y_test_total - pred_total_ridge)^2) / sum((y_test_total - mean(y_test_total))^2)

cat("\n📊 Ridge Performance on total_UPDRS:\n")
cat("   ➤ MSE =", round(mse_ridge, 4), "\n")
cat("   ➤ R²  =", round(r2_ridge, 4), "\n")

# ================================
# 6. Print Ridge Model Formula
# ================================
cat("\n🧮 Final Ridge Model Formula:\n")

# Extract non-zero coefficients (including intercept)
coefs_ridge <- as.matrix(coef(ridge_total_final))
nonzero_ridge <- coefs_ridge[which(coefs_ridge != 0), , drop = FALSE]

# Build the formula
terms_ridge <- c()
for (i in 1:nrow(nonzero_ridge)) {
  name <- rownames(nonzero_ridge)[i]
  value <- round(nonzero_ridge[i, 1], 4)
  if (name == "(Intercept)") {
    intercept_ridge <- value
  } else {
    terms_ridge <- c(terms_ridge, paste0(value, " * ", name))
  }
}

cat("y = ", intercept_ridge, " + ", paste(terms_ridge, collapse = " + "), "\n")

# ===============================================================
# STEP 5.3 – ElasticNet Regression for total_UPDRS
# ===============================================================

library(glmnet)
library(Metrics)
library(ggplot2)

cat("\n📌 Training ElasticNet model for target: total_UPDRS\n")

# ================================
# 1. Data Preparation
# ================================

# Exclude non-predictive columns and the target variable
X_train_matrix <- as.matrix(X_train[, !colnames(X_train) %in% c("is_outlier", "total_UPDRS", "motor_UPDRS")])
X_test_matrix  <- as.matrix(X_test[, !colnames(X_test) %in% c("is_outlier", "total_UPDRS", "motor_UPDRS")])

# Display matrix dimensions
cat("\nDimension of X_train_matrix:", dim(X_train_matrix), "\n")
cat("Dimension of X_test_matrix:", dim(X_test_matrix), "\n")

# ================================
# 2. Alpha Tuning with Cross-validation
# ================================
set.seed(123)
alphas <- seq(0, 1, by = 0.1)
mse_list <- c()
models_list <- list()

# Cross-validation for each alpha value
for (a in alphas) {
  cv_fit <- cv.glmnet(
    x = X_train_matrix,
    y = y_train_total,
    alpha = a,
    nfolds = 10,
    standardize = FALSE
  )
  mse_list <- c(mse_list, min(cv_fit$cvm))
  models_list[[as.character(a)]] <- cv_fit
}

# Identify best alpha and corresponding model
best_alpha <- alphas[which.min(mse_list)]
best_model <- models_list[[as.character(best_alpha)]]
best_lambda <- best_model$lambda.min

cat("\n✅ Best alpha:", best_alpha, "\n")
cat("✅ Best lambda:", best_lambda, "\n")

# ================================
# 3. Plot MSE vs Alpha
# ================================
df_alpha <- data.frame(alpha = alphas, MSE = mse_list)

ggplot(df_alpha, aes(x = alpha, y = MSE)) +
  geom_line(color = "lightgreen", linewidth = 2) +
  geom_point(color = "lightgreen", size = 3) +
  geom_vline(xintercept = best_alpha, color = "red", linetype = "dashed", linewidth = 1.2) +
  labs(
    title = "ElasticNet – Cross-Validated MSE vs Alpha",
    subtitle = paste("Optimal alpha =", best_alpha),
    x = "Alpha (mix L1 and L2)",
    y = "Mean Squared Error (MSE)"
  ) +
  theme_minimal()

# ================================
# 4. Final Model and Prediction
# ================================
elastic_total_final <- glmnet(
  x = X_train_matrix,
  y = y_train_total,
  alpha = best_alpha,
  lambda = best_lambda,
  standardize = FALSE
)

pred_total_elastic <- predict(elastic_total_final, s = best_lambda, newx = X_test_matrix)

# ================================
# 5. Performance Metrics
# ================================
mse_elastic  <- mse(y_test_total, pred_total_elastic)
rmse_elastic <- sqrt(mse_elastic)
mae_elastic  <- mae(y_test_total, pred_total_elastic)
r2_elastic   <- 1 - sum((y_test_total - pred_total_elastic)^2) / sum((y_test_total - mean(y_test_total))^2)

cat("\n📊 ElasticNet Performance on total_UPDRS:\n")
cat("   ➤ MSE  =", round(mse_elastic, 4), "\n")
cat("   ➤ RMSE =", round(rmse_elastic, 4), "\n")
cat("   ➤ MAE  =", round(mae_elastic, 4), "\n")
cat("   ➤ R²   =", round(r2_elastic, 4), "\n")

# ================================
# 6. Selected Coefficients
# ================================
coef_elastic_total <- coef(elastic_total_final)
selected_vars <- rownames(coef_elastic_total)[which(coef_elastic_total != 0)]

cat("\n📌 Selected variables:\n")
print(selected_vars)

# ================================
# 7. Final Model Formula
# ================================

cat("\n🧮 Final ElasticNet Model Formula:\n")

coefs <- as.matrix(coef_elastic_total)
nonzero_coefs <- coefs[which(coefs != 0), , drop = FALSE]

terms <- c()
for (i in 1:nrow(nonzero_coefs)) {
  name <- rownames(nonzero_coefs)[i]
  value <- round(nonzero_coefs[i, 1], 4)
  if (name == "(Intercept)") {
    intercept <- value
  } else {
    terms <- c(terms, paste0(value, " * ", name))
  }
}

cat("y = ", intercept, " + ", paste(terms, collapse = " + "), "\n")


# ===============================================================
# STEP – Forward Selection for Multiple Linear Regression
# Target: total_UPDRS
# ===============================================================

library(leaps)

cat("\n📌 Starting forward variable selection for total_UPDRS\n")

# ---------------------------------------------------------------
# 1. Preparing the dataset for selection
# ---------------------------------------------------------------
forward_data_total <- cbind(total_UPDRS = y_total_clean,
                            X_clean_scaled[, !names(X_clean_scaled) %in% c("is_outlier", "motor_UPDRS")])
colnames(forward_data_total) <- make.names(colnames(forward_data_total))

x_matrix_total <- model.matrix(total_UPDRS ~ ., data = forward_data_total)[, -1]
y_vector_total <- forward_data_total$total_UPDRS

# ---------------------------------------------------------------
# 2. Forward selection (nvmax = 15)
# ---------------------------------------------------------------
forward_model_total <- regsubsets(x = x_matrix_total, y = y_vector_total, nvmax = 15, method = "forward")
summary_forward_total <- summary(forward_model_total)

# ---------------------------------------------------------------
# 3. Select best model using BIC
# ---------------------------------------------------------------
best_bic_index_total <- which.min(summary_forward_total$bic)
cat("\n✅ Best model selected according to BIC (index =", best_bic_index_total, ")\n")

selected_vars_total <- names(which(summary_forward_total$which[best_bic_index_total, ]))[-1]
cat("📌 Selected variables:\n")
print(selected_vars_total)

# ---------------------------------------------------------------
# 4. Build final linear model
# ---------------------------------------------------------------
formula_forward_total <- as.formula(
  paste("total_UPDRS ~", paste(selected_vars_total, collapse = " + "))
)
lm_forward_total <- lm(formula_forward_total, data = forward_data_total)

cat("\n📊 Summary of the final model:\n")
print(summary(lm_forward_total))

# ---------------------------------------------------------------
# 5. Diagnostic plots saved as PNG
# ---------------------------------------------------------------
png("EDA_plot/forward_diagnostics_total_UPDRS.png", width = 1000, height = 1000, res = 150)
par(mfrow = c(2, 2))
plot(lm_forward_total)
dev.off()
cat("📁 Diagnostic plots saved to: EDA_plot/forward_diagnostics_total_UPDRS.png\n")

# ================================
# 6. Final Model Formula
# ================================

cat("\n🧮 Final Model Formula (Forward Selection):\n")

coefs_forward <- coef(lm_forward_total)
intercept_forward <- round(coefs_forward[1], 4)

terms_forward <- c()
for (i in 2:length(coefs_forward)) {
  name <- names(coefs_forward)[i]
  value <- round(coefs_forward[i], 4)
  terms_forward <- c(terms_forward, paste0(value, " * ", name))
}

cat("y = ", intercept_forward, " + ", paste(terms_forward, collapse = " + "), "\n")

# Print 95% confidence intervals for coefficients
confint(lm_forward_total, level = 0.95)


# ===============================================================
# STEP – Backward Selection for Multiple Linear Regression
# Target: total_UPDRS
# ===============================================================

library(leaps)

cat("\n📌 Starting backward variable selection for total_UPDRS\n")

# ---------------------------------------------------------------
# 1. Preparing the dataset for selection
# ---------------------------------------------------------------
backward_data_total <- cbind(total_UPDRS = y_total_clean,
                             X_clean_scaled[, !names(X_clean_scaled) %in% c("is_outlier", "motor_UPDRS")])
colnames(backward_data_total) <- make.names(colnames(backward_data_total))

x_matrix_total <- model.matrix(total_UPDRS ~ ., data = backward_data_total)[, -1]
y_vector_total <- backward_data_total$total_UPDRS

# ---------------------------------------------------------------
# 2. Backward selection (nvmax = 15)
# ---------------------------------------------------------------
backward_model_total <- regsubsets(x = x_matrix_total, y = y_vector_total, nvmax = 15, method = "backward")
summary_backward_total <- summary(backward_model_total)

# ---------------------------------------------------------------
# 3. Select best model using BIC
# ---------------------------------------------------------------
best_bic_index_backward_total <- which.min(summary_backward_total$bic)
cat("\n✅ Best model selected according to BIC (index =", best_bic_index_backward_total, ")\n")

selected_vars_backward_total <- names(which(summary_backward_total$which[best_bic_index_backward_total, ]))[-1]
cat("📌 Selected variables:\n")
print(selected_vars_backward_total)

# ---------------------------------------------------------------
# 4. Build final linear model
# ---------------------------------------------------------------
formula_backward_total <- as.formula(
  paste("total_UPDRS ~", paste(selected_vars_backward_total, collapse = " + "))
)
lm_backward_total <- lm(formula_backward_total, data = backward_data_total)

cat("\n📊 Summary of the final model:\n")
print(summary(lm_backward_total))

# ---------------------------------------------------------------
# 5. Diagnostic plots saved as PNG
# ---------------------------------------------------------------
png("EDA_plot/backward_diagnostics_total_UPDRS.png", width = 1000, height = 1000, res = 150)
par(mfrow = c(2, 2))
plot(lm_backward_total)
dev.off()
cat("📁 Diagnostic plots saved to: EDA_plot/backward_diagnostics_total_UPDRS.png\n")

# ================================
# 6. Final Model Formula
# ================================

cat("\n🧮 Final Model Formula (Backward Selection):\n")

coefs_backward <- coef(lm_backward_total)
intercept_backward <- round(coefs_backward[1], 4)

terms_backward <- c()
for (i in 2:length(coefs_backward)) {
  name <- names(coefs_backward)[i]
  value <- round(coefs_backward[i], 4)
  terms_backward <- c(terms_backward, paste0(value, " * ", name))
}

cat("y = ", intercept_backward, " + ", paste(terms_backward, collapse = " + "), "\n")

# Print 95% confidence intervals for coefficients
confint(lm_backward_total, level = 0.95)




# ===============================================================
# STEP – Principal Component Regression (PCR) on total_UPDRS
# ===============================================================

if (!require("pls")) install.packages("pls")
if (!require("ggplot2")) install.packages("ggplot2")

library(pls)
library(Metrics)
library(caret)
library(ggplot2)

cat("\n📌 Training PCR model for target: total_UPDRS\n")

# ---------------------------------------------------------------
# 1. Prepare dataset: remove non-predictive columns
# ---------------------------------------------------------------
X_train_pcr_total <- X_train[, !colnames(X_train) %in% c("is_outlier", "total_UPDRS", "motor_UPDRS")]
X_test_pcr_total  <- X_test[,  !colnames(X_test)  %in% c("is_outlier", "total_UPDRS", "motor_UPDRS")]

X_train_pcr_total <- as.data.frame(X_train_pcr_total)
X_test_pcr_total  <- as.data.frame(X_test_pcr_total)

# ---------------------------------------------------------------
# 2. Fit PCR with 10-fold cross-validation
# ---------------------------------------------------------------
set.seed(123)

pcr_model_total <- pcr(
  y_train_total ~ .,  
  data = X_train_pcr_total, 
  scale = FALSE,
  validation = "CV", 
  segments = 10
)

# ---------------------------------------------------------------
# 3. Select optimal number of components (min PRESS)
# ---------------------------------------------------------------
opt_ncomp_total <- which.min(pcr_model_total$validation$PRESS)
cat("\n✅ Optimal number of components:", opt_ncomp_total, "\n")

# ---------------------------------------------------------------
# 4. Predict on test set using optimal components
# ---------------------------------------------------------------
pred_pcr_total <- predict(pcr_model_total, newdata = X_test_pcr_total, ncomp = opt_ncomp_total)
pred_pcr_total <- as.numeric(pred_pcr_total)

# ---------------------------------------------------------------
# 5. Evaluate performance manually
# ---------------------------------------------------------------
mse_pcr_total  <- mse(y_test_total, pred_pcr_total)
rmse_pcr_total <- sqrt(mse_pcr_total)
mae_pcr_total  <- mae(y_test_total, pred_pcr_total)
r2_pcr_total   <- 1 - sum((y_test_total - pred_pcr_total)^2) / sum((y_test_total - mean(y_test_total))^2)

cat("\n📊 PCR Model Performance on total_UPDRS:\n")
cat("   ➤ MSE  =", round(mse_pcr_total, 4), "\n")
cat("   ➤ RMSE =", round(rmse_pcr_total, 4), "\n")
cat("   ➤ MAE  =", round(mae_pcr_total, 4), "\n")
cat("   ➤ R²   =", round(r2_pcr_total, 4), "\n")

# ---------------------------------------------------------------
# 6A. Plot: MSEP vs Components
# ---------------------------------------------------------------
msep_vals_total <- MSEP(pcr_model_total, estimate = "CV")$val[1, 1, ]
df_msep_total <- data.frame(Components = seq_along(msep_vals_total), MSEP = msep_vals_total)

plot_msep <- ggplot(df_msep_total, aes(x = Components, y = MSEP)) +
  geom_line(color = "darkblue", linewidth = 1) +
  geom_point(color = "lightblue", size = 2) +
  geom_vline(xintercept = opt_ncomp_total, linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title = "PCR – Cross-Validated MSE vs Number of Components (total_UPDRS)",
    subtitle = paste("Optimal number of components:", opt_ncomp_total),
    x = "Number of Principal Components",
    y = "MSEP"
  ) +
  theme_minimal()

print(plot_msep)
ggsave("PLOT/pcr_msep_total_UPDRS.png", plot = plot_msep, width = 8, height = 6)

# ---------------------------------------------------------------
# 6B. Plot: Top 10 Loadings on First Principal Component
# ---------------------------------------------------------------
load_mat_total <- loadings(pcr_model_total)
pc1_loadings_total <- as.vector(load_mat_total[, 1])
names(pc1_loadings_total) <- rownames(load_mat_total)

top10_total <- sort(abs(pc1_loadings_total), decreasing = TRUE)[1:10]
top10_named_total <- pc1_loadings_total[names(top10_total)]

df_load_total <- data.frame(
  Variable = factor(names(top10_named_total), levels = rev(names(top10_named_total))),
  Loading = top10_named_total
)

plot_loadings <- ggplot(df_load_total, aes(x = Variable, y = Loading)) +
  geom_bar(stat = "identity", fill = "lightblue", color = "darkblue") +
  coord_flip() +
  labs(
    title = "Top 10 Loadings on First Principal Component (total_UPDRS)",
    x = "Variable",
    y = "Loading Value"
  ) +
  theme_minimal()

print(plot_loadings)
ggsave("PLOT/pcr_loadings_total_UPDRS.png", plot = plot_loadings, width = 8, height = 6)

# ================================
# 7. Final Model Formula (PCR)
# ================================

cat("\n🧮 Final Model Formula (PCR in terms of principal components):\n")

coefs_pcr <- coef(pcr_model_total, ncomp = opt_ncomp_total)
coefs_vec <- as.vector(coefs_pcr)
intercept_pcr <- round(coefs_vec[1], 4)

terms_pcr <- c()
for (i in 2:(opt_ncomp_total + 1)) {
  coef_val <- round(coefs_vec[i], 4)
  terms_pcr <- c(terms_pcr, paste0(coef_val, " * PC", i - 1))
}

cat("y = ", intercept_pcr, " + ", paste(terms_pcr, collapse = " + "), "\n")

# ===============================================================
# STEP – PLS Regression for total_UPDRS
# ===============================================================

library(pls)
library(ggplot2)
library(Metrics)

cat("\n📌 Training PLS model for target: total_UPDRS\n")

# ---------------------------------------------------------------
# 1. Data Preparation
# ---------------------------------------------------------------
X_train_pls_total <- X_train[, !colnames(X_train) %in% c("is_outlier", "motor_UPDRS", "total_UPDRS")]
X_test_pls_total  <- X_test[,  !colnames(X_test) %in% c("is_outlier", "motor_UPDRS", "total_UPDRS")]

train_pls_total <- data.frame(X_train_pls_total, total_UPDRS = y_train_total)
test_pls_total  <- data.frame(X_test_pls_total,  total_UPDRS = y_test_total)

# ---------------------------------------------------------------
# 2. Fit with Cross-Validation
# ---------------------------------------------------------------
set.seed(123)
pls_fit_total <- plsr(total_UPDRS ~ ., data = train_pls_total, scale = TRUE, validation = "CV")

# ---------------------------------------------------------------
# 3. Validation Plot
# ---------------------------------------------------------------
dev.new()
validationplot(pls_fit_total, val.type = "MSEP", main = "PLS – CV Error by # of Components (total_UPDRS)")
ggsave("PLOT/pls_cv_pls_total_UPDRS.png", plot = last_plot(), width = 8, height = 6, dpi = 300)

# ---------------------------------------------------------------
# 4. Optimal Number of Components
# ---------------------------------------------------------------
ncomp_opt_total <- which.min(RMSEP(pls_fit_total)$val[1,1,-1])
cat("\n✅ Optimal number of components:", ncomp_opt_total, "\n")

# ---------------------------------------------------------------
# 5. Prediction and Metrics
# ---------------------------------------------------------------
pred_total_pls <- predict(pls_fit_total, newdata = test_pls_total, ncomp = ncomp_opt_total)

mse_total_pls  <- mse(y_test_total, pred_total_pls)
rmse_total_pls <- sqrt(mse_total_pls)
mae_total_pls  <- mae(y_test_total, pred_total_pls)
ss_res_total <- sum((y_test_total - as.numeric(pred_total_pls))^2)
ss_tot_total <- sum((y_test_total - mean(y_test_total))^2)
r2_total_pls <- 1 - ss_res_total / ss_tot_total

cat("\n📊 PLS Model Performance on total_UPDRS:\n")
cat("   ➤ MSE  =", round(mse_total_pls, 4), "\n")
cat("   ➤ RMSE =", round(rmse_total_pls, 4), "\n")
cat("   ➤ MAE  =", round(mae_total_pls, 4), "\n")
cat("   ➤ R²   =", round(r2_total_pls, 4), "\n")

# ---------------------------------------------------------------
# 6. Coefficients
# ---------------------------------------------------------------
pls_coeffs_total <- coef(pls_fit_total, ncomp = ncomp_opt_total)
cat("\n📌 Coefficients of PLS (selected components – total_UPDRS):\n")
print(pls_coeffs_total)

# ---------------------------------------------------------------
# 7. Final Model Formula (PLS components)
# ---------------------------------------------------------------

cat("\n🧮 Approximate PLS Model Formula (in terms of latent components):\n")

coefs_pls <- drop(coef(pls_fit_total, ncomp = ncomp_opt_total))
intercept_pls <- round(mean(y_train_total), 4)

pls_scores <- scores(pls_fit_total)[, 1:ncomp_opt_total]
colnames(pls_scores) <- paste0("PLS", 1:ncomp_opt_total)

pls_component_model <- lm(y_train_total ~ pls_scores)

intercept_est <- round(coef(pls_component_model)[1], 4)
coeffs_latent <- coef(pls_component_model)[-1]
terms_pls_latent <- paste0(round(coeffs_latent, 4), " * PLS", 1:length(coeffs_latent))

cat("y = ", intercept_est, " + ", paste(terms_pls_latent, collapse = " + "), "\n")


# ===============================================================
# STEP – KNN Regression for total_UPDRS
# ===============================================================

library(caret)
library(Metrics)

cat("\n📌 Training KNN model for target: total_UPDRS\n")

# ---------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------
X_train_knn_total <- X_train[, !colnames(X_train) %in% c("is_outlier", "motor_UPDRS", "total_UPDRS")]
X_test_knn_total  <- X_test[,  !colnames(X_test)  %in% c("is_outlier", "motor_UPDRS", "total_UPDRS")]

y_train_knn_total <- y_train_total
y_test_knn_total  <- y_test_total

# ---------------------------------------------------------------
# 2. Search for Best K
# ---------------------------------------------------------------
mse_list_total <- numeric(20)
for (k in 1:20) {
  model_k <- knnreg(X_train_knn_total, y_train_knn_total, k = k)
  pred_k  <- predict(model_k, X_test_knn_total)
  mse_list_total[k] <- mean((y_test_knn_total - pred_k)^2)
}

best_k_total <- which.min(mse_list_total)
cat("✅ Best k:", best_k_total, "with MSE =", round(mse_list_total[best_k_total], 4), "\n")

# ---------------------------------------------------------------
# 3. Final Model Fit with Best K
# ---------------------------------------------------------------
knn_final_model_total <- knnreg(X_train_knn_total, y_train_knn_total, k = best_k_total)
pred_knn_total <- predict(knn_final_model_total, X_test_knn_total)

# ---------------------------------------------------------------
# 4. Evaluation Metrics
# ---------------------------------------------------------------
mse_knn_total  <- mse(y_test_knn_total, pred_knn_total)
rmse_knn_total <- sqrt(mse_knn_total)
mae_knn_total  <- mae(y_test_knn_total, pred_knn_total)
ss_res_total   <- sum((y_test_knn_total - pred_knn_total)^2)
ss_tot_total   <- sum((y_test_knn_total - mean(y_test_knn_total))^2)
r2_knn_total   <- 1 - ss_res_total / ss_tot_total

cat("\n📊 KNN Model Performance on total_UPDRS:\n")
cat("   ➤ MSE  =", round(mse_knn_total, 4), "\n")
cat("   ➤ RMSE =", round(rmse_knn_total, 4), "\n")
cat("   ➤ MAE  =", round(mae_knn_total, 4), "\n")
cat("   ➤ R²   =", round(r2_knn_total, 4), "\n")

# ---------------------------------------------------------------
# 5. Plot Comparison with Linear Regression
# ---------------------------------------------------------------
lm_model_total <- lm(y_train_knn_total ~ ., data = data.frame(X_train_knn_total))
pred_lm_total  <- predict(lm_model_total, newdata = data.frame(X_test_knn_total))

x <- 1:length(y_test_knn_total)
dev.new()
plot(x, y_test_knn_total, col = 'red', type = 'l', lty = 2, lwd = 2,
     xlab = "Sample Index", ylab = "total_UPDRS", main = "KNN vs Linear Regression – total_UPDRS")
lines(x, pred_knn_total, col = 'blue', lwd = 2)
lines(x, pred_lm_total,  col = 'green', lwd = 2)
legend('topright', legend = c('True', 'KNN', 'Linear'),
       col = c('red', 'blue', 'green'), lty = 1, lwd = 2)
png("EDA_plot/knn_vs_lm_total_UPDRS.png", width = 1200, height = 800, res = 150)
plot(x, y_test_knn_total, col = 'red', type = 'l', lty = 2, lwd = 2,
     xlab = "Sample Index", ylab = "total_UPDRS", main = "KNN vs Linear Regression – total_UPDRS")
lines(x, pred_knn_total, col = 'blue', lwd = 2)
lines(x, pred_lm_total,  col = 'green', lwd = 2)
legend('topright', legend = c('True', 'KNN', 'Linear'),
       col = c('red', 'blue', 'green'), lty = 1, lwd = 2)
dev.off()



# ===============================================================
# STEP – Lasso Regression for motor_UPDRS (with CV plot)
# ===============================================================

library(glmnet)
library(Metrics)
library(ggplot2)
library(caret)

cat("\n📌 Training LASSO model for target: motor_UPDRS\n")

# ---------------------------------------------------------------
# 1. Data Preparation
# ---------------------------------------------------------------
X_train_matrix <- as.matrix(X_train[, !colnames(X_train) %in% c("is_outlier", "total_UPDRS", "motor_UPDRS")])
X_test_matrix  <- as.matrix(X_test[, !colnames(X_test) %in% c("is_outlier", "total_UPDRS", "motor_UPDRS")])

y_train_motor <- y_train_motor
y_test_motor  <- y_test_motor

# ---------------------------------------------------------------
# 2. Cross-validation with LASSO
# ---------------------------------------------------------------
set.seed(123)
lasso_motor_cv <- cv.glmnet(
  x = X_train_matrix,
  y = y_train_motor,
  alpha = 1,
  nfolds = 10,
  standardize = FALSE
)

# ---------------------------------------------------------------
# 3. Plot CV
# ---------------------------------------------------------------
log_lambda <- log(lasso_motor_cv$lambda)
mse_cv     <- lasso_motor_cv$cvm
stderr     <- lasso_motor_cv$cvsd
best_lambda_motor <- lasso_motor_cv$lambda.min
best_log_lambda   <- log(best_lambda_motor)

df_cv <- data.frame(log_lambda, mse_cv, stderr)

ggplot(df_cv, aes(x = log_lambda, y = mse_cv)) +
  geom_point(color = "lightgreen", size = 2) +
  geom_line(color = "lightgreen", linewidth = 1.5) +
  geom_errorbar(aes(ymin = mse_cv - stderr, ymax = mse_cv + stderr),
                width = 0.2, color = "gray50") +
  geom_vline(xintercept = best_log_lambda, linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title = "LASSO – Cross-Validated MSE vs Log(Lambda)",
    subtitle = paste("Optimal lambda =", round(best_lambda_motor, 5)),
    x = "log(Lambda)",
    y = "Mean Squared Error (MSE)"
  ) +
  theme_minimal()

# ---------------------------------------------------------------
# 4. Final model with optimal lambda
# ---------------------------------------------------------------
lasso_motor_final <- glmnet(
  x = X_train_matrix,
  y = y_train_motor,
  alpha = 1,
  lambda = best_lambda_motor,
  standardize = FALSE
)

# ---------------------------------------------------------------
# 5. Prediction and metrics
# ---------------------------------------------------------------
pred_motor_lasso <- predict(lasso_motor_final, s = best_lambda_motor, newx = X_test_matrix)

mse_motor_lasso  <- mse(y_test_motor, pred_motor_lasso)
rmse_motor_lasso <- sqrt(mse_motor_lasso)
mae_motor_lasso  <- mae(y_test_motor, pred_motor_lasso)

ss_res <- sum((y_test_motor - as.numeric(pred_motor_lasso))^2)
ss_tot <- sum((y_test_motor - mean(y_test_motor))^2)
r2_motor_lasso <- 1 - ss_res / ss_tot

cat("\n📊 LASSO Performance on motor_UPDRS:\n")
cat("   ➤ MSE  =", round(mse_motor_lasso, 4), "\n")
cat("   ➤ RMSE =", round(rmse_motor_lasso, 4), "\n")
cat("   ➤ MAE  =", round(mae_motor_lasso, 4), "\n")
cat("   ➤ R²   =", round(r2_motor_lasso, 4), "\n")

# ---------------------------------------------------------------
# 6. Selected variables
# ---------------------------------------------------------------
coef_lasso_motor <- coef(lasso_motor_final)
selected_motor_features <- rownames(coef_lasso_motor)[which(coef_lasso_motor != 0)]

cat("\n📌 Selected variables by LASSO (motor_UPDRS):\n")
print(selected_motor_features)

# ---------------------------------------------------------------
# 7. Print model formula
# ---------------------------------------------------------------
coef_df <- as.data.frame(as.matrix(coef_lasso_motor))
colnames(coef_df) <- "Coefficient"
coef_df$Variable <- rownames(coef_df)

coef_df <- coef_df[coef_df$Coefficient != 0, ]
coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]

intercept <- coef_df$Coefficient[coef_df$Variable == "(Intercept)"]
terms <- coef_df[coef_df$Variable != "(Intercept)", ]
formula_string <- paste0("motor_UPDRS = ", round(intercept, 4))

for (i in 1:nrow(terms)) {
  coef_val <- round(terms$Coefficient[i], 4)
  var_name <- terms$Variable[i]
  sign <- ifelse(coef_val >= 0, " + ", " - ")
  formula_string <- paste0(formula_string, sign, abs(coef_val), "*", var_name)
}

cat("\n🧮 Selected LASSO model formula:\n")
cat(formula_string, "\n")


# ===============================================================
# STEP – Ridge Regression for motor_UPDRS (with CV plot)
# ===============================================================

library(glmnet)
library(Metrics)
library(ggplot2)
library(caret)

cat("\n📌 Training Ridge model for target: motor_UPDRS\n")

# ---------------------------------------------------------------
# 1. Data Preparation
# ---------------------------------------------------------------
X_train_matrix <- as.matrix(X_train[, !colnames(X_train) %in% c("is_outlier", "total_UPDRS", "motor_UPDRS")])
X_test_matrix  <- as.matrix(X_test[, !colnames(X_test) %in% c("is_outlier", "total_UPDRS", "motor_UPDRS")])

# ---------------------------------------------------------------
# 2. Cross-validation with Ridge (alpha = 0)
# ---------------------------------------------------------------
set.seed(123)

ridge_motor_cv <- cv.glmnet(
  x = X_train_matrix,
  y = y_train_motor,
  alpha = 0,
  nfolds = 10,
  standardize = FALSE
)

# Extract CV info
log_lambda_ridge <- log(ridge_motor_cv$lambda)
mse_cv_ridge     <- ridge_motor_cv$cvm
stderr_ridge     <- ridge_motor_cv$cvsd
best_lambda_ridge <- ridge_motor_cv$lambda.min
best_log_lambda_ridge <- log(best_lambda_ridge)

df_cv_ridge <- data.frame(log_lambda_ridge, mse_cv_ridge, stderr_ridge)

# ---------------------------------------------------------------
# 3. CV Plot
# ---------------------------------------------------------------
p_ridge_cv <- ggplot(df_cv_ridge, aes(x = log_lambda_ridge, y = mse_cv_ridge)) +
  geom_point(color = "lightblue", size = 2) +
  geom_line(color = "steelblue", linewidth = 1.5) +
  geom_errorbar(aes(ymin = mse_cv_ridge - stderr_ridge, ymax = mse_cv_ridge + stderr_ridge),
                width = 0.2, color = "gray50") +
  geom_vline(xintercept = best_log_lambda_ridge, linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title = "Ridge – Cross-Validated MSE vs Log(Lambda)",
    subtitle = paste("Optimal lambda =", round(best_lambda_ridge, 5)),
    x = "log(Lambda)",
    y = "Mean Squared Error (MSE)"
  ) +
  theme_minimal()

print(p_ridge_cv)

ggsave("EDA_plot/ridge_cv_motor_UPDRS.png", plot = p_ridge_cv, width = 7, height = 5, dpi = 300)

# ---------------------------------------------------------------
# 4. Final model
# ---------------------------------------------------------------
ridge_motor_final <- glmnet(
  x = X_train_matrix,
  y = y_train_motor,
  alpha = 0,
  lambda = best_lambda_ridge,
  standardize = FALSE
)

# ---------------------------------------------------------------
# 5. Prediction and metrics
# ---------------------------------------------------------------
pred_motor_ridge <- predict(ridge_motor_final, s = best_lambda_ridge, newx = X_test_matrix)

mse_motor_ridge  <- mse(y_test_motor, pred_motor_ridge)
rmse_motor_ridge <- sqrt(mse_motor_ridge)
mae_motor_ridge  <- mae(y_test_motor, pred_motor_ridge)

ss_res <- sum((y_test_motor - as.numeric(pred_motor_ridge))^2)
ss_tot <- sum((y_test_motor - mean(y_test_motor))^2)
r2_motor_ridge <- 1 - ss_res / ss_tot

cat("\n📊 Ridge Performance on motor_UPDRS:\n")
cat("   ➤ MSE  =", round(mse_motor_ridge, 4), "\n")
cat("   ➤ RMSE =", round(rmse_motor_ridge, 4), "\n")
cat("   ➤ MAE  =", round(mae_motor_ridge, 4), "\n")
cat("   ➤ R²   =", round(r2_motor_ridge, 4), "\n")

# ---------------------------------------------------------------
# 6. Print model formula
# ---------------------------------------------------------------
coef_ridge_motor <- coef(ridge_motor_final)
coef_df <- as.data.frame(as.matrix(coef_ridge_motor))
colnames(coef_df) <- "Coefficient"
coef_df$Variable <- rownames(coef_df)

coef_df <- coef_df[coef_df$Coefficient != 0, ]
coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]

intercept <- coef_df$Coefficient[coef_df$Variable == "(Intercept)"]
terms <- coef_df[coef_df$Variable != "(Intercept)", ]
formula_string <- paste0("motor_UPDRS = ", round(intercept, 4))

for (i in 1:nrow(terms)) {
  coef_val <- round(terms$Coefficient[i], 4)
  var_name <- terms$Variable[i]
  sign <- ifelse(coef_val >= 0, " + ", " - ")
  formula_string <- paste0(formula_string, sign, abs(coef_val), "*", var_name)
}

cat("\n🧮 Selected RIDGE model formula:\n")
cat(formula_string, "\n")




# ===============================================================
# STEP – ElasticNet Regression for motor_UPDRS (with CV plot)
# ===============================================================

library(glmnet)
library(Metrics)
library(ggplot2)

cat("\n📌 Training ElasticNet model for target: motor_UPDRS\n")

# ---------------------------------------------------------------
# 1. Data Preparation
# ---------------------------------------------------------------
X_train_matrix <- as.matrix(X_train[, !colnames(X_train) %in% c("is_outlier", "total_UPDRS", "motor_UPDRS")])
X_test_matrix  <- as.matrix(X_test[, !colnames(X_test) %in% c("is_outlier", "total_UPDRS", "motor_UPDRS")])

y_train_motor <- y_train_motor
y_test_motor  <- y_test_motor

# ---------------------------------------------------------------
# 2. Grid search on alpha (ElasticNet)
# ---------------------------------------------------------------
alphas <- seq(0, 1, by = 0.1)
mse_list <- c()
models_list <- list()

set.seed(123)
for (a in alphas) {
  cv_fit <- cv.glmnet(
    x = X_train_matrix,
    y = y_train_motor,
    alpha = a,
    nfolds = 10,
    standardize = FALSE
  )
  mse_list <- c(mse_list, min(cv_fit$cvm))
  models_list[[as.character(a)]] <- cv_fit
}

best_alpha <- alphas[which.min(mse_list)]
best_model <- models_list[[as.character(best_alpha)]]
best_lambda <- best_model$lambda.min

cat("\n✅ Best alpha:", best_alpha, "\n")
cat("✅ Best lambda:", best_lambda, "\n")

# ---------------------------------------------------------------
# 3. CV Plot
# ---------------------------------------------------------------
log_lambda_elastic <- log(best_model$lambda)
mse_cv_elastic     <- best_model$cvm
stderr_elastic     <- best_model$cvsd
best_log_lambda_elastic <- log(best_lambda)

df_cv_elastic <- data.frame(log_lambda_elastic, mse_cv_elastic, stderr_elastic)

p_elastic_cv <- ggplot(df_cv_elastic, aes(x = log_lambda_elastic, y = mse_cv_elastic)) +
  geom_point(color = "orchid", size = 2) +
  geom_line(color = "purple", linewidth = 1.5) +
  geom_errorbar(aes(ymin = mse_cv_elastic - stderr_elastic, ymax = mse_cv_elastic + stderr_elastic),
                width = 0.2, color = "gray50") +
  geom_vline(xintercept = best_log_lambda_elastic, linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title = "ElasticNet – Cross-Validated MSE vs Log(Lambda)",
    subtitle = paste("Alpha =", best_alpha, "| Optimal lambda =", round(best_lambda, 5)),
    x = "log(Lambda)",
    y = "Mean Squared Error (MSE)"
  ) +
  theme_minimal()

print(p_elastic_cv)
ggsave("EDA_plot/elasticnet_cv_motor_UPDRS.png", plot = p_elastic_cv, width = 7, height = 5, dpi = 300)

# ---------------------------------------------------------------
# 4. Final model and prediction
# ---------------------------------------------------------------
elastic_motor_final <- glmnet(
  x = X_train_matrix,
  y = y_train_motor,
  alpha = best_alpha,
  lambda = best_lambda,
  standardize = FALSE
)

pred_motor_elastic <- predict(elastic_motor_final, s = best_lambda, newx = X_test_matrix)

# ---------------------------------------------------------------
# 5. Performance metrics
# ---------------------------------------------------------------
mse_motor_elastic  <- mse(y_test_motor, pred_motor_elastic)
rmse_motor_elastic <- sqrt(mse_motor_elastic)
mae_motor_elastic  <- mae(y_test_motor, pred_motor_elastic)

ss_res <- sum((y_test_motor - as.numeric(pred_motor_elastic))^2)
ss_tot <- sum((y_test_motor - mean(y_test_motor))^2)
r2_motor_elastic <- 1 - ss_res / ss_tot

cat("\n📊 ElasticNet Performance on motor_UPDRS:\n")
cat("   ➤ MSE  =", round(mse_motor_elastic, 4), "\n")
cat("   ➤ RMSE =", round(rmse_motor_elastic, 4), "\n")
cat("   ➤ MAE  =", round(mae_motor_elastic, 4), "\n")
cat("   ➤ R²   =", round(r2_motor_elastic, 4), "\n")

# ---------------------------------------------------------------
# 6. Print selected coefficients
# ---------------------------------------------------------------
coef_elastic_motor <- coef(elastic_motor_final)
coef_df <- as.data.frame(as.matrix(coef_elastic_motor))
colnames(coef_df) <- "Coefficient"
coef_df$Variable <- rownames(coef_df)

coef_df <- coef_df[coef_df$Coefficient != 0, ]
coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]

intercept <- coef_df$Coefficient[coef_df$Variable == "(Intercept)"]
terms <- coef_df[coef_df$Variable != "(Intercept)", ]
formula_string <- paste0("motor_UPDRS = ", round(intercept, 4))

for (i in 1:nrow(terms)) {
  coef_val <- round(terms$Coefficient[i], 4)
  var_name <- terms$Variable[i]
  sign <- ifelse(coef_val >= 0, " + ", " - ")
  formula_string <- paste0(formula_string, sign, abs(coef_val), "*", var_name)
}

cat("\n🧮 Selected ElasticNet model formula:\n")
cat(formula_string, "\n")








# ===============================================================
# STEP – Forward Selection for Multiple Linear Regression
# Target: motor_UPDRS
# ===============================================================

library(leaps)

cat("\n📌 Starting forward variable selection for motor_UPDRS\n")

# ---------------------------------------------------------------
# 1. Preparing the dataset for selection
# ---------------------------------------------------------------
forward_data <- cbind(motor_UPDRS = y_motor_clean,
                      X_clean_scaled[, !names(X_clean_scaled) %in% c("is_outlier", "total_UPDRS")])
colnames(forward_data) <- make.names(colnames(forward_data))

x_matrix <- model.matrix(motor_UPDRS ~ ., data = forward_data)[, -1]
y_vector <- forward_data$motor_UPDRS

# ---------------------------------------------------------------
# 2. Forward selection (nvmax = 15)
# ---------------------------------------------------------------
forward_model <- regsubsets(x = x_matrix, y = y_vector, nvmax = 15, method = "forward")
summary_forward <- summary(forward_model)

# ---------------------------------------------------------------
# 3. Select best model using BIC
# ---------------------------------------------------------------
best_bic_index <- which.min(summary_forward$bic)
cat("\n✅ Best model selected according to BIC (index =", best_bic_index, ")\n")

selected_vars <- names(which(summary_forward$which[best_bic_index, ]))[-1]
cat("📌 Selected variables:\n")
print(selected_vars)

# ---------------------------------------------------------------
# 4. Build final linear model
# ---------------------------------------------------------------
formula_forward <- as.formula(
  paste("motor_UPDRS ~", paste(selected_vars, collapse = " + "))
)
lm_forward <- lm(formula_forward, data = forward_data)

cat("\n📊 Summary of the final model:\n")
print(summary(lm_forward))

# ---------------------------------------------------------------
# 5. Diagnostic plots saved as PNG
# ---------------------------------------------------------------
png("EDA_plot/forward_diagnostics.png", width = 1000, height = 1000, res = 150)
par(mfrow = c(2, 2))
plot(lm_forward)
dev.off()
cat("📁 Diagnostic plots saved to: EDA_plot/forward_diagnostics.png\n")

# ---------------------------------------------------------------
# 6. Print final model formula
# ---------------------------------------------------------------
intercept_forward <- coef(lm_forward)[1]
terms_forward <- coef(lm_forward)[-1]

# Create formula terms as a string
terms_string <- paste0(round(terms_forward, 4), " * ", names(terms_forward))

# Build and print the formula
cat("y = ", round(intercept_forward, 4), " + ", paste(terms_string, collapse = " + "), "\n")

cat("\n📊 95% Confidence Intervals – Forward Selection:\n")
print(confint(lm_forward, level = 0.95))



# ===============================================================
# STEP – Backward Selection for Multiple Linear Regression
# Target: motor_UPDRS
# ===============================================================

library(leaps)

cat("\n📌 Starting backward variable selection for motor_UPDRS\n")

# ---------------------------------------------------------------
# 1. Preparing the dataset for selection
# ---------------------------------------------------------------
backward_data <- cbind(motor_UPDRS = y_motor_clean,
                       X_clean_scaled[, !names(X_clean_scaled) %in% c("is_outlier", "total_UPDRS")])
colnames(backward_data) <- make.names(colnames(backward_data))

x_matrix <- model.matrix(motor_UPDRS ~ ., data = backward_data)[, -1]
y_vector <- backward_data$motor_UPDRS

# ---------------------------------------------------------------
# 2. Backward selection (nvmax = 15)
# ---------------------------------------------------------------
backward_model <- regsubsets(x = x_matrix, y = y_vector, nvmax = 15, method = "backward")
summary_backward <- summary(backward_model)

# ---------------------------------------------------------------
# 3. Select best model using BIC
# ---------------------------------------------------------------
best_bic_index_backward <- which.min(summary_backward$bic)
cat("\n✅ Best model selected according to BIC (index =", best_bic_index_backward, ")\n")

selected_vars_backward <- names(which(summary_backward$which[best_bic_index_backward, ]))[-1]
cat("📌 Selected variables:\n")
print(selected_vars_backward)

# ---------------------------------------------------------------
# 4. Build final linear model
# ---------------------------------------------------------------
formula_backward <- as.formula(
  paste("motor_UPDRS ~", paste(selected_vars_backward, collapse = " + "))
)
lm_backward <- lm(formula_backward, data = backward_data)

cat("\n📊 Summary of the final model:\n")
print(summary(lm_backward))

# ---------------------------------------------------------------
# 5. Diagnostic plots saved as PNG
# ---------------------------------------------------------------
png("EDA_plot/backward_diagnostics.png", width = 1000, height = 1000, res = 150)
par(mfrow = c(2, 2))
plot(lm_backward)
dev.off()
cat("📁 Diagnostic plots saved to: EDA_plot/backward_diagnostics.png\n")

# ---------------------------------------------------------------
# 6. Print final model formula
# ---------------------------------------------------------------
intercept_backward <- coef(lm_backward)[1]
terms_backward <- coef(lm_backward)[-1]
terms_string_backward <- paste0(round(terms_backward, 4), " * ", names(terms_backward))

cat("\n🧮 Final model formula (Backward Selection):\n")
cat("y = ", round(intercept_backward, 4), " + ", paste(terms_string_backward, collapse = " + "), "\n")

cat("\n📊 95% Confidence Intervals – Backward Selection:\n")
print(confint(lm_backward, level = 0.95))



# ===============================================================
# STEP – Principal Component Regression (PCR) on motor_UPDRS
# ===============================================================

if (!require("pls")) install.packages("pls")
if (!require("ggplot2")) install.packages("ggplot2")

library(pls)
library(Metrics)
library(caret)
library(ggplot2)

cat("\n📌 Training PCR model for target: motor_UPDRS\n")

# ---------------------------------------------------------------
# 1. Prepare dataset: remove non-predictive columns
# ---------------------------------------------------------------
X_train_pcr_motor <- X_train[, !colnames(X_train) %in% c("is_outlier", "motor_UPDRS", "total_UPDRS")]
X_test_pcr_motor  <- X_test[,  !colnames(X_test)  %in% c("is_outlier", "motor_UPDRS", "total_UPDRS")]

X_train_pcr_motor <- as.data.frame(X_train_pcr_motor)
X_test_pcr_motor  <- as.data.frame(X_test_pcr_motor)

# ---------------------------------------------------------------
# 2. Fit PCR with 10-fold cross-validation
# ---------------------------------------------------------------
set.seed(123)
# Split
set.seed(42)
train_index <- sample(seq_len(nrow(X_clean_scaled)), size = 0.8 * nrow(X_clean_scaled))

# Subset in modo sincronizzato
X_train <- X_clean_scaled[train_index, ]
X_test  <- X_clean_scaled[-train_index, ]

y_train_total <- y_total_clean[train_index]
y_test_total  <- y_total_clean[-train_index]

y_train_motor <- y_motor_clean[train_index]
y_test_motor  <- y_motor_clean[-train_index]

# Crea un dataframe unico con le feature + target
df_train_motor_pcr <- X_train_pcr_motor
df_train_motor_pcr$y_train_motor <- y_train_motor

# Ora puoi usare la formula
pcr_model_motor <- pcr(
  y_train_motor ~ .,  
  data = df_train_motor_pcr, 
  scale = FALSE,
  validation = "CV", 
  segments = 10
)


# ---------------------------------------------------------------
# 3. Select optimal number of components (min PRESS)
# ---------------------------------------------------------------
opt_ncomp_motor <- which.min(pcr_model_motor$validation$PRESS)
cat("\n✅ Optimal number of components:", opt_ncomp_motor, "\n")

# ---------------------------------------------------------------
# 4. Predict on test set using optimal components
# ---------------------------------------------------------------
pred_pcr_motor <- predict(pcr_model_motor, newdata = X_test_pcr_motor, ncomp = opt_ncomp_motor)
pred_pcr_motor <- as.numeric(pred_pcr_motor)

# ---------------------------------------------------------------
# 5. Evaluate performance
# ---------------------------------------------------------------
mse_pcr_motor  <- mse(y_test_motor, pred_pcr_motor)
rmse_pcr_motor <- sqrt(mse_pcr_motor)
mae_pcr_motor  <- mae(y_test_motor, pred_pcr_motor)
r2_pcr_motor   <- 1 - sum((y_test_motor - pred_pcr_motor)^2) / sum((y_test_motor - mean(y_test_motor))^2)

cat("\n📊 PCR Model Performance on motor_UPDRS:\n")
cat("   ➤ MSE  =", round(mse_pcr_motor, 4), "\n")
cat("   ➤ RMSE =", round(rmse_pcr_motor, 4), "\n")
cat("   ➤ MAE  =", round(mae_pcr_motor, 4), "\n")
cat("   ➤ R²   =", round(r2_pcr_motor, 4), "\n")

# ---------------------------------------------------------------
# 6A. Plot: MSEP vs Components
# ---------------------------------------------------------------
msep_vals_motor <- MSEP(pcr_model_motor, estimate = "CV")$val[1, 1, ]
df_msep_motor <- data.frame(Components = seq_along(msep_vals_motor), MSEP = msep_vals_motor)

plot_msep_motor <- ggplot(df_msep_motor, aes(x = Components, y = MSEP)) +
  geom_line(color = "darkblue", linewidth = 1) +
  geom_point(color = "lightblue", size = 2) +
  geom_vline(xintercept = opt_ncomp_motor, linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title = "PCR – Cross-Validated MSE vs Number of Components (motor_UPDRS)",
    subtitle = paste("Optimal number of components:", opt_ncomp_motor),
    x = "Number of Principal Components",
    y = "MSEP"
  ) +
  theme_minimal()

print(plot_msep_motor)
ggsave("PLOT/pcr_msep_motor_UPDRS.png", plot = plot_msep_motor, width = 8, height = 6)

# ---------------------------------------------------------------
# 6B. Plot: Top 10 Loadings on First Principal Component
# ---------------------------------------------------------------
load_mat_motor <- loadings(pcr_model_motor)
pc1_loadings_motor <- as.vector(load_mat_motor[, 1])
names(pc1_loadings_motor) <- rownames(load_mat_motor)

top10_motor <- sort(abs(pc1_loadings_motor), decreasing = TRUE)[1:10]
top10_named_motor <- pc1_loadings_motor[names(top10_motor)]

df_load_motor <- data.frame(
  Variable = factor(names(top10_named_motor), levels = rev(names(top10_named_motor))),
  Loading = top10_named_motor
)

plot_loadings_motor <- ggplot(df_load_motor, aes(x = Variable, y = Loading)) +
  geom_bar(stat = "identity", fill = "lightblue", color = "darkblue") +
  coord_flip() +
  labs(
    title = "Top 10 Loadings on First Principal Component (motor_UPDRS)",
    x = "Variable",
    y = "Loading Value"
  ) +
  theme_minimal()

print(plot_loadings_motor)
ggsave("PLOT/pcr_loadings_motor_UPDRS.png", plot = plot_loadings_motor, width = 8, height = 6)

# ================================
# 7. Print final PCR model formula
# ================================
cat("\n🧮 Formula del modello PCR (in termini di componenti principali):\n")

coefs_pcr_motor <- coef(pcr_model_motor, ncomp = opt_ncomp_motor)
coefs_vec_motor <- as.vector(coefs_pcr_motor)
intercept_motor <- round(coefs_vec_motor[1], 4)

terms_motor <- c()
for (i in 2:(opt_ncomp_motor + 1)) {
  coef_val <- round(coefs_vec_motor[i], 4)
  terms_motor <- c(terms_motor, paste0(coef_val, " * PC", i - 1))
}

cat("y = ", intercept_motor, " + ", paste(terms_motor, collapse = " + "), "\n")


# ===============================================================
# STEP – PLS Regression for motor_UPDRS
# ===============================================================

library(pls)
library(ggplot2)
library(Metrics)

cat("\n📌 Training PLS model for target: motor_UPDRS\n")

# ---------------------------------------------------------------
# 1. Data Preparation
# ---------------------------------------------------------------
# Remove non-predictive variables
X_train_pls_motor <- X_train[, !colnames(X_train) %in% c("is_outlier", "motor_UPDRS", "total_UPDRS")]
X_test_pls_motor  <- X_test[,  !colnames(X_test)  %in% c("is_outlier", "motor_UPDRS", "total_UPDRS")]

# Combine X and y for the pls package
train_pls_motor <- data.frame(X_train_pls_motor, motor_UPDRS = y_train_motor)
test_pls_motor  <- data.frame(X_test_pls_motor,  motor_UPDRS = y_test_motor)

# ---------------------------------------------------------------
# 2. Fit with cross-validation
# ---------------------------------------------------------------
set.seed(123)
pls_fit_motor <- plsr(motor_UPDRS ~ ., data = train_pls_motor, scale = TRUE, validation = "CV")

# ---------------------------------------------------------------
# 3. Validation plot (CV)
# ---------------------------------------------------------------
dev.new()
validationplot(pls_fit_motor, val.type = "MSEP", main = "PLS – CV Error by # of Components (motor_UPDRS)")
ggsave("PLOT/pls_cv_motor_UPDRS.png", plot = last_plot(), width = 8, height = 6, dpi = 300)

# ---------------------------------------------------------------
# 4. Optimal number of components
# ---------------------------------------------------------------
ncomp_opt_motor <- which.min(RMSEP(pls_fit_motor)$val[1, 1, -1])  # Exclude component 0
cat("\n✅ Optimal number of components:", ncomp_opt_motor, "\n")

# ---------------------------------------------------------------
# 5. Prediction and performance metrics
# ---------------------------------------------------------------
pred_motor_pls <- predict(pls_fit_motor, newdata = test_pls_motor, ncomp = ncomp_opt_motor)

mse_motor_pls  <- mse(y_test_motor, pred_motor_pls)
rmse_motor_pls <- sqrt(mse_motor_pls)
mae_motor_pls  <- mae(y_test_motor, pred_motor_pls)
ss_res_motor <- sum((y_test_motor - as.numeric(pred_motor_pls))^2)
ss_tot_motor <- sum((y_test_motor - mean(y_test_motor))^2)
r2_motor_pls <- 1 - ss_res_motor / ss_tot_motor

cat("\n📊 PLS Model Performance on motor_UPDRS:\n")
cat("   ➤ MSE  =", round(mse_motor_pls, 4), "\n")
cat("   ➤ RMSE =", round(rmse_motor_pls, 4), "\n")
cat("   ➤ MAE  =", round(mae_motor_pls, 4), "\n")
cat("   ➤ R²   =", round(r2_motor_pls, 4), "\n")

# ---------------------------------------------------------------
# 6. Coefficients
# ---------------------------------------------------------------
pls_coeffs_motor <- coef(pls_fit_motor, ncomp = ncomp_opt_motor)
cat("\n📌 Coefficients of PLS (selected components – motor_UPDRS):\n")
print(pls_coeffs_motor)

# ---------------------------------------------------------------
# 7. Final PLS model formula (latent components)
# ---------------------------------------------------------------

cat("\n🧮 Approximate PLS model formula (in terms of latent components):\n")

# Extract coefficient vector
coefs_pls_motor <- drop(coef(pls_fit_motor, ncomp = ncomp_opt_motor))

# Retrieve intercept (mean of y in training set)
intercept_motor_pls <- round(mean(y_train_motor), 4)

# Component scores
pls_scores_motor <- scores(pls_fit_motor)[, 1:ncomp_opt_motor]
colnames(pls_scores_motor) <- paste0("PLS", 1:ncomp_opt_motor)

# Regression: y ~ PLS components
pls_component_model_motor <- lm(y_train_motor ~ pls_scores_motor)

# Construct formula
intercept_est_motor <- round(coef(pls_component_model_motor)[1], 4)
coeffs_latent_motor <- coef(pls_component_model_motor)[-1]
terms_pls_latent_motor <- paste0(round(coeffs_latent_motor, 4), " * PLS", 1:length(coeffs_latent_motor))

# Print final formula
cat("y = ", intercept_est_motor, " + ", paste(terms_pls_latent_motor, collapse = " + "), "\n")



# ===============================================================
# STEP – KNN Regression for motor_UPDRS
# ===============================================================

library(caret)
library(Metrics)

cat("\n📌 Training KNN for target: motor_UPDRS\n")

# ---------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------
X_train_knn_motor <- X_train[, !colnames(X_train) %in% c("is_outlier", "motor_UPDRS", "total_UPDRS")]
X_test_knn_motor  <- X_test[,  !colnames(X_test)  %in% c("is_outlier", "motor_UPDRS", "total_UPDRS")]

y_train_knn_motor <- y_train_motor
y_test_knn_motor  <- y_test_motor

# ---------------------------------------------------------------
# 2. Best K Search
# ---------------------------------------------------------------
mse_list_motor <- numeric(20)
for (k in 1:20) {
  model_k <- knnreg(X_train_knn_motor, y_train_knn_motor, k = k)
  pred_k  <- predict(model_k, X_test_knn_motor)
  mse_list_motor[k] <- mean((y_test_knn_motor - pred_k)^2)
}

best_k_motor <- which.min(mse_list_motor)
cat("✅ Best k:", best_k_motor, "with MSE =", round(mse_list_motor[best_k_motor], 4), "\n")

# ---------------------------------------------------------------
# 3. Final Fit with Best K
# ---------------------------------------------------------------
knn_final_model_motor <- knnreg(X_train_knn_motor, y_train_knn_motor, k = best_k_motor)
pred_knn_motor <- predict(knn_final_model_motor, X_test_knn_motor)

# ---------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------
mse_knn_motor  <- mse(y_test_knn_motor, pred_knn_motor)
rmse_knn_motor <- sqrt(mse_knn_motor)
mae_knn_motor  <- mae(y_test_knn_motor, pred_knn_motor)
ss_res_motor   <- sum((y_test_knn_motor - pred_knn_motor)^2)
ss_tot_motor   <- sum((y_test_knn_motor - mean(y_test_knn_motor))^2)
r2_knn_motor   <- 1 - ss_res_motor / ss_tot_motor

cat("\n📊 KNN Performance on motor_UPDRS:\n")
cat("   ➤ MSE  =", round(mse_knn_motor, 4), "\n")
cat("   ➤ RMSE =", round(rmse_knn_motor, 4), "\n")
cat("   ➤ MAE  =", round(mae_knn_motor, 4), "\n")
cat("   ➤ R²   =", round(r2_knn_motor, 4), "\n")

# ---------------------------------------------------------------
# 5. Plot Comparison with Linear Regression
# ---------------------------------------------------------------
lm_model_motor <- lm(y_train_knn_motor ~ ., data = data.frame(X_train_knn_motor))
pred_lm_motor  <- predict(lm_model_motor, newdata = data.frame(X_test_knn_motor))

x <- 1:length(y_test_knn_motor)
dev.new()
plot(x, y_test_knn_motor, col = 'red', type = 'l', lty = 2, lwd = 2,
     xlab = "Sample Index", ylab = "motor_UPDRS", main = "KNN vs Linear Regression – motor_UPDRS")
lines(x, pred_knn_motor, col = 'blue', lwd = 2)
lines(x, pred_lm_motor,  col = 'green', lwd = 2)
legend('topright', legend = c('True', 'KNN', 'Linear'),
       col = c('red', 'blue', 'green'), lty = 1, lwd = 2)
png("EDA_plot/knn_vs_lm_motor_UPDRS.png", width = 1200, height = 800, res = 150)
plot(x, y_test_knn_motor, col = 'red', type = 'l', lty = 2, lwd = 2,
     xlab = "Sample Index", ylab = "motor_UPDRS", main = "KNN vs Linear Regression – motor_UPDRS")
lines(x, pred_knn_motor, col = 'blue', lwd = 2)
lines(x, pred_lm_motor,  col = 'green', lwd = 2)
legend('topright', legend = c('True', 'KNN', 'Linear'),
       col = c('red', 'blue', 'green'), lty = 1, lwd = 2)
dev.off()