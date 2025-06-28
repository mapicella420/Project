
# ================================
# STEP 1 – Load Required Libraries
# ================================

# Optional: install missing packages
# install.packages("corrplot")
# install.packages("reshape2")

# Core libraries
library(readr)       # For reading CSV files
library(dplyr)       # For data wrangling and manipulation
library(ggplot2)     # For plotting and data visualization
library(corrplot)    # For visualizing correlation matrices
library(reshape2)    # For reshaping data frames (used in heatmaps)
library(scales)      # For improving axis formatting and labels
library(rlang)       # For symbolic variable evaluation in tidyverse functions

# Statistical tools
library(car)         # For regression diagnostics (e.g., VIF)
library(stats)       # For basic statistical functions

# ===========================================
# STEP 2 – Load and Inspect the Dataset
# ===========================================

# Set working directory (adjust if needed)
setwd("C:/Users/Mario/Desktop/Regressione")

# Load the dataset from CSV
df <- read_csv("parkinsons_updrs.csv")

# Display structure and summary of the dataset
str(df)              # Data structure
names(df)            # Column names
summary(df)          # Summary statistics

# Check for missing values (NA)
table(is.na(df))     # Total number of missing values

# Preview first and last rows
head(df)             # First 5 rows
tail(df)             # Last 5 rows

# Create a working copy for cleaning and processing
df_clean <- df

# ===============================================================
# STEP 3 – Data Cleaning (Missing values, Duplicates, Constants)
# ===============================================================

## 3.1 – Check for missing values
any(is.na(df_clean))              # Returns TRUE if there are any NA values
colSums(is.na(df_clean))          # Count of NA values per column

# The dataset is expected to have no missing values (based on documentation).
# If needed, use the following:
# df_clean <- na.omit(df_clean)   # Option: remove rows with any NA
# Or apply appropriate imputation methods

## 3.2 – Identify and remove duplicate rows
duplicates <- df_clean[duplicated(df_clean), ]
n_duplicates <- nrow(duplicates)
cat("Number of duplicate rows:", n_duplicates, "\n")

# Remove duplicated rows
df_clean <- df_clean[!duplicated(df_clean), ]

## 3.3 – Remove non-informative (constant) features
# Identify variables with zero variance
constant_vars <- sapply(df_clean, function(x) length(unique(x)) == 1)
non_informative <- names(df_clean)[constant_vars]
cat("Non-informative variables:", non_informative, "\n")

# Drop constant variables
df_clean <- df_clean[ , !constant_vars]

## 3.4 – Convert binary variables to labeled factors
# Example: convert 'sex' column from 0/1 to Male/Female
df_clean$sex <- factor(df_clean$sex, levels = c(0, 1), labels = c("Male", "Female"))

# Verify conversion
str(df_clean$sex)
summary(df_clean$sex)

# Note: If modeling tools (e.g., glmnet) require dummy variables,
# use model.matrix() for one-hot encoding later.

## 3.5 – Final structure check after cleaning
cat("Final dataset dimensions:", dim(df_clean), "\n")
str(df_clean)
summary(df_clean)



# ===============================================================
# STEP 3 – Exploratory Data Analysis (EDA)
#
# GOAL: Visualize each numeric feature to assess its distribution,
# variability, and presence of outliers.
# ===============================================================

# 🔹 Select numeric predictors (excluding target variables and ID)
numeric_vars <- df_clean %>%
  dplyr::select(where(is.numeric)) %>%
  dplyr::select(-total_UPDRS, -motor_UPDRS, -`subject#`)

# Extract variable names for plotting
var_names <- colnames(numeric_vars)

# ===============================================================
# 3.1 – Histogram + Density Plots for each numeric variable
# ===============================================================

for (var in var_names) {
  var_sym <- sym(var)  # Convert variable name to symbol for tidy evaluation
  print(
    ggplot(df_clean, aes(x = !!var_sym)) +
      geom_histogram(aes(y = after_stat(density)), bins = 40,
                     fill = "lightblue", color = "white", alpha = 0.7) +
      geom_density(fill = "lightblue", alpha = 0.5, color = "blue", linewidth = 1) +
      theme_minimal() +
      labs(title = paste("Histogram + Density of", var),
           x = var, y = "Density")
  )
}

# ===============================================================
# 3.2 – Boxplots for all numeric variables
# ===============================================================

# Create folder to store plots if not already present
if (!dir.exists("EDA_plot")) {
  dir.create("EDA_plot")
}

# Custom function to generate and save annotated boxplots
plot_custom_boxplot <- function(x, varname = "Variable") {
  filename_safe <- gsub("[^A-Za-z0-9_]", "_", varname)  # Sanitize filename
  
  # Save plot to PNG
  png(filename = paste0("EDA_plot/", filename_safe, "_boxplot.png"),
      width = 800, height = 600, res = 150)
  
  boxplot(x,
          notch = TRUE,
          col = "lightblue",
          border = "blue4",
          main = paste("Boxplot:", varname),
          ylab = varname)
  
  # Add labels for boxplot statistics
  stats <- boxplot.stats(x)$stats
  labels <- c("minimum", "1st quartile", "median", "3rd quartile", "maximum")
  for (i in 1:5) {
    text(x = 1.2, y = stats[i], labels = labels[i],
         pos = 4, col = "blue4", cex = 0.8, font = 3)
  }
  
  dev.off()  # Close the graphics device
}

# Loop through numeric variables and generate boxplots
for (var in names(numeric_vars)) {
  x <- numeric_vars[[var]]
  plot_custom_boxplot(x, var)
}

# ===============================================================
# 3.3 – Rug Plot + Density Plot
# ===============================================================

for (var in var_names) {
  var_sym <- sym(var)
  print(
    ggplot(df_clean, aes(x = !!var_sym)) +
      geom_rug(sides = "b", alpha = 0.3, color = "steelblue") +
      geom_density(fill = "lightblue", alpha = 0.4, color = "blue4", linewidth = 1) +
      theme_minimal() +
      labs(title = paste("Rug Plot + Density of", var),
           x = var, y = "Density")
  )
}

# ===============================================================
# STEP 3.4 – Distribution of Target Variables (total_UPDRS and motor_UPDRS)
# ===============================================================

# Create output directory if not present
if (!dir.exists("EDA_plot")) {
  dir.create("EDA_plot")
}

# Loop through both targets and create visualizations
for (target_name in targets) {
  target <- df_clean[[target_name]]
  
  # Histogram + Density plot
  p1 <- ggplot(df_clean, aes(x = !!sym(target_name))) +
    geom_histogram(aes(y = after_stat(density)), bins = 40,
                   fill = "lightblue", color = "white", alpha = 0.7) +
    geom_density(fill = "skyblue", alpha = 0.4, color = "blue4", linewidth = 1) +
    theme_minimal() +
    labs(title = paste("Histogram + Density of", target_name),
         x = target_name, y = "Density")
  
  # Save plot
  ggsave(paste0("EDA_plot/", target_name, "_density.png"),
         plot = p1, width = 6, height = 4, dpi = 300)
  
  # Q-Q Plot (Normality check)
  png(paste0("EDA_plot/", target_name, "_qqplot.png"), width = 800, height = 600)
  qqnorm(target, main = paste("Q-Q Plot of", target_name), col = "blue4")
  qqline(target, col = "red", lwd = 2)
  dev.off()
  
  # Shapiro-Wilk test for normality (sampled if too large)
  shapiro_result <- shapiro.test(sample(target, min(5000, length(target))))
  print(shapiro_result)
}

# ===============================================================
# STEP 3.5 – Boxplots of Targets by Sex
# ===============================================================

for (target_name in targets) {
  p2 <- ggplot(df_clean, aes(x = sex, y = !!sym(target_name), fill = sex)) +
    geom_boxplot(outlier.shape = 16, outlier.color = "black") +
    scale_fill_manual(values = c("skyblue", "lightblue")) +
    theme_minimal() +
    labs(title = paste(target_name, "by Sex"),
         x = "Sex", y = target_name)
  
  # Save boxplot
  ggsave(paste0("EDA_plot/", target_name, "_by_Sex.png"),
         plot = p2, width = 6, height = 4, dpi = 300)
}
# ===============================================================
# STEP 3.5 – Correlation Matrix (for total_UPDRS and motor_UPDRS)
# ===============================================================

# ======================================
# 1. Select numeric variables (including both targets)
# ======================================
cor_data <- df_clean %>%
  dplyr::select(where(is.numeric)) %>%
  dplyr::select(-`subject#`)  # remove ID column

# ======================================
# 2. Compute Pearson correlation matrix
# ======================================
cor_matrix <- cor(cor_data, method = "pearson")

# ======================================
# 3. Visualize full correlation matrix (optional overview)
# ======================================
corrplot(cor_matrix,
         method = "ellipse",
         type = "upper",
         order = "hclust",
         tl.col = "black",
         tl.cex = 0.8,
         col = colorRampPalette(c("skyblue", "white", "blue4"))(100))

# ======================================
# 4. Analyze correlations for each target separately
# ======================================
targets <- c("total_UPDRS", "motor_UPDRS")

for (target in targets) {
  cat("\n=====================================\n")
  cat(paste("▶ Correlations with target:", target, "\n"))
  cat("=====================================\n")
  
  target_corr <- cor_matrix[target, ]
  target_corr <- target_corr[!names(target_corr) %in% targets]
  target_corr <- sort(target_corr, decreasing = TRUE)
  
  cat("\nTop 10 positively correlated features:\n")
  print(head(round(target_corr, 3), 10))
  
  cat("\nTop 10 negatively correlated features:\n")
  print(tail(round(target_corr, 3), 10))
}


# ===============================================================
# STEP 3.6 – Multicollinearity Analysis (VIF)
# ===============================================================

library(dplyr)
library(car)

targets <- c("total_UPDRS", "motor_UPDRS")

for (target_name in targets) {
  cat("\n=============================\n")
  cat(paste("▶ VIF analysis for target:", target_name, "\n"))
  cat("=============================\n")
  
  predictors <- df_clean %>%
    dplyr::select(where(is.numeric)) %>%
    dplyr::select(-all_of(targets), -`subject#`)
  
  if ("sex" %in% colnames(df_clean)) {
    predictors$sex <- df_clean$sex
  }
  
  safe_names <- paste0("`", names(predictors), "`")
  formula <- as.formula(paste(target_name, "~", paste(safe_names, collapse = " + ")))
  
  lm_model <- lm(formula, data = df_clean)
  vif_values <- vif(lm_model)
  vif_sorted <- sort(vif_values, decreasing = TRUE)
  
  print(round(vif_sorted, 2))
  cat("\n⚠️ Variables with VIF > 5:\n")
  print(round(vif_sorted[vif_sorted > 5], 2))
}


# ===============================================================
# STEP 3.7 – Outlier Detection
# ===============================================================

library(ggplot2)
library(MASS)
library(rlang)
library(car)

dir.create("EDA_plot", showWarnings = FALSE)

numeric_vars <- df_clean %>%
  dplyr::select(where(is.numeric)) %>%
  dplyr::select(-c(`total_UPDRS`, `motor_UPDRS`, `subject#`))

for (var in names(numeric_vars)) {
  x <- numeric_vars[[var]]
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  outliers <- which(x < lower_bound | x > upper_bound)
  
  cat(paste0("🔍 Variable: ", var, "\n"))
  cat("   ➤ Outliers detected: ", length(outliers), " (", round(length(outliers)/length(x)*100, 2), "%)\n\n")
  
  var_sym <- rlang::sym(var)
  p <- ggplot(df_clean, aes(y = !!var_sym)) +
    geom_boxplot(fill = "lightblue", color = "blue4", outlier.colour = "red", outlier.shape = 16) +
    theme_minimal() +
    labs(title = paste("Boxplot + Outliers of", var), y = var)
  
  filename <- paste0("EDA_plot/", gsub("[^a-zA-Z0-9]", "_", var), "_outlier_boxplot.png")
  ggsave(filename = filename, plot = p, width = 6, height = 5, dpi = 300)
}

# Mahalanobis Distance
X_mahal <- df_clean %>%
  select(where(is.numeric)) %>%
  select(-c(`subject#`, `total_UPDRS`, `motor_UPDRS`))

md <- mahalanobis(X_mahal, colMeans(X_mahal), cov(X_mahal))
df_clean$Mahalanobis_Dist <- md

cutoff <- qchisq(p = 0.975, df = ncol(X_mahal))
df_clean$Mahalanobis_Outlier <- md > cutoff

p1 <- ggplot(df_clean, aes(x = Mahalanobis_Dist)) +
  geom_histogram(bins = 80, fill = "#4DA8DA", color = "white", alpha = 0.85) +
  geom_vline(xintercept = cutoff, color = "red", linetype = "dashed", linewidth = 1) +
  theme_minimal() +
  labs(title = "Mahalanobis Distance – Full Histogram",
       x = "Mahalanobis Distance", y = "Count")
ggsave("EDA_plot/mahalanobis_full.png", p1, width = 8, height = 5.5, dpi = 300)

# Studentized Residuals
for (target in targets) {
  cat("\n===============================\n")
  cat(paste("Studentized Residuals for:", target, "\n"))
  cat("===============================\n")
  
  predictors <- df_clean %>%
    dplyr::select(where(is.numeric)) %>%
    dplyr::select(-c(`subject#`, `total_UPDRS`, `motor_UPDRS`))
  
  if ("sex" %in% colnames(df_clean)) {
    predictors$sex <- df_clean$sex
  }
  
  formula <- as.formula(
    paste0("`", target, "` ~ ", paste0("`", names(predictors), "`", collapse = " + "))
  )
  
  model <- lm(formula, data = df_clean)
  stud_res <- rstudent(model)
  plot_df <- data.frame(Sample = 1:length(stud_res), Residual = stud_res)
  
  p <- ggplot(plot_df, aes(x = Sample, y = Residual)) +
    geom_point(alpha = 0.5, color = "black") +
    geom_hline(yintercept = c(-2, 2), linetype = "dashed", color = "red", linewidth = 0.8) +
    geom_point(data = subset(plot_df, abs(Residual) > 2), color = "red", alpha = 0.7) +
    theme_minimal() +
    labs(title = paste("Outlier Detection via Studentized Residuals on", target),
         x = "Sample Index", y = "Studentized Residuals")
  
  filename <- paste0("EDA_plot/studentized_", target, ".png")
  ggsave(filename, p, width = 7, height = 5, dpi = 300)
}


# ===============================================================
# STEP 3.8 – Visualization (Pairplot, PCA, Heatmap)
# ===============================================================

if (!require(GGally)) install.packages("GGally")
library(GGally)
library(ggplot2)

# Pair plot for selected features
plot_data <- df_clean %>%
  dplyr::select(sex, total_UPDRS, motor_UPDRS, DFA, PPE, Shimmer, NHR)

pair_plot_colored <- ggpairs(
  data = plot_data,
  columns = 2:7,
  aes(color = sex, alpha = 0.5),
  title = "Pair Plot: UPDRS Scores and Key Features by Sex",
  upper = list(continuous = wrap("cor", size = 3)),
  lower = list(continuous = wrap("points", alpha = 0.4, size = 0.8)),
  diag = list(continuous = wrap("densityDiag", alpha = 0.5))
)

ggsave("EDA_plot/pairplot_colored_by_sex.png", plot = pair_plot_colored, width = 12, height = 10, dpi = 300)

# PCA
pca_data <- df_clean %>%
  dplyr::select(where(is.numeric)) %>%
  dplyr::select(-c(`subject#`, `total_UPDRS`, `motor_UPDRS`))

pca_scaled <- scale(pca_data)
pca_result <- prcomp(pca_scaled)
pca_df <- data.frame(pca_result$x[, 1:2])
colnames(pca_df) <- c("PC1", "PC2")

for (target in c("total_UPDRS", "motor_UPDRS")) {
  pca_df$target <- df_clean[[target]]
  
  p <- ggplot(pca_df, aes(x = PC1, y = PC2, color = target)) +
    geom_point(alpha = 0.7, size = 2) +
    scale_color_gradient(low = "skyblue", high = "darkblue") +
    theme_minimal() +
    labs(title = paste("PCA Scatter Plot colored by", target),
         x = "Principal Component 1", y = "Principal Component 2")
  
  filename <- paste0("EDA_plot/pca_colored_", target, ".png")
  ggsave(filename, plot = p, width = 8, height = 6, dpi = 300)
}

# Heatmap
if (!require(pheatmap)) install.packages("pheatmap")
library(pheatmap)

heatmap_data <- as.data.frame(pca_scaled[1:15, ])
rownames(heatmap_data) <- paste0("Sample_", 1:15)

png("EDA_plot/heatmap_top15_standardized.png", width = 1200, height = 1000, res = 150)
pheatmap(heatmap_data,
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         show_rownames = TRUE,
         color = colorRampPalette(c("skyblue", "white", "blue4"))(15),
         main = "Heatmap of Standardized Features")
dev.off()



# ===============================================================
# STEP 4.1 – Target Selection and Verification
#
# GOAL: Check, summarize and visualize both regression targets.
# Targets: total_UPDRS and motor_UPDRS
# ===============================================================

library(ggplot2)

# Define target variables
targets <- c("total_UPDRS", "motor_UPDRS")

# Loop through each target
for (target in targets) {
  cat("\n=====================================\n")
  cat(paste("📌 Checking target variable:", target, "\n"))
  cat("=====================================\n")
  
  # --- Check existence and type ---
  if (!target %in% colnames(df_clean)) {
    stop(paste("❌ Target", target, "not found in dataset."))
  }
  if (!is.numeric(df_clean[[target]])) {
    stop(paste("❌ Target", target, "is not numeric."))
  }
  cat(paste("✅", target, "exists and is numeric.\n"))
  
  # --- Summary statistics ---
  cat("\n📊 Summary statistics:\n")
  print(summary(df_clean[[target]]))
  
  # --- Histogram with density overlay ---
  plot <- ggplot(df_clean, aes(x = .data[[target]])) +
    geom_histogram(bins = 40, fill = "steelblue", color = "white", alpha = 0.7) +
    geom_density(aes(y = after_stat(density)), color = "darkred", linewidth = 1) +
    theme_minimal() +
    labs(title = paste("Distribution of:", target),
         x = target,
         y = "Density")
  
  print(plot)
  Sys.sleep(1)  # Optional: delay to view plots sequentially
}

# --- Extract target vectors for later use ---
y_total <- df_clean$total_UPDRS
y_motor <- df_clean$motor_UPDRS

cat("\n📦 Target vectors extracted:\n")
cat("• y_total → "); str(y_total)
cat("• y_motor → "); str(y_motor)


# ===============================================================
# STEP 4.2 – Feature Selection & Cleaning
#
# GOAL: Prepare predictors by removing:
# - ID fields
# - both target variables
# - Mahalanobis distance (if present)
# - zero-variance features
# ===============================================================

library(dplyr)

# Select only numeric columns
X_clean <- df_clean %>% select_if(is.numeric)
cat("🔍 Selected numeric columns:", ncol(X_clean), "\n")

# Remove non-predictive columns
columns_to_remove <- c("subject#", "total_UPDRS", "motor_UPDRS", "Mahalanobis_Dist")
columns_to_remove <- intersect(columns_to_remove, colnames(X_clean))

if (length(columns_to_remove) > 0) {
  X_clean <- X_clean %>% select(-all_of(columns_to_remove))
  cat("🧹 Removed non-predictive columns:", paste(columns_to_remove, collapse = ", "), "\n")
} else {
  cat("✅ No non-predictive columns to remove.\n")
}

# Remove zero-variance predictors
zero_var_cols <- names(X_clean)[apply(X_clean, 2, sd) == 0]

if (length(zero_var_cols) > 0) {
  X_clean <- X_clean %>% select(-all_of(zero_var_cols))
  cat("⚠️ Removed", length(zero_var_cols), "zero-variance variables:\n")
  print(zero_var_cols)
} else {
  cat("✅ No zero-variance variables found.\n")
}

cat("✅ Final predictor matrix: ", nrow(X_clean), "rows ×", ncol(X_clean), "columns\n")


# ===============================================================
# STEP 4.3 – Outlier Handling (Mahalanobis Distance)
#
# GOAL:
# • Add 'is_outlier' flag to the predictor matrix
# • Remove extreme outliers based on Mahalanobis distance
# • Prepare X_clean, y_total_clean and y_motor_clean
# ===============================================================

# Recalculate Mahalanobis distance if not present
if (!"Mahalanobis_Dist" %in% colnames(df_clean)) {
  message("ℹ Recalculating Mahalanobis Distance...")
  X_mahal <- df_clean %>%
    dplyr::select(where(is.numeric)) %>%
    dplyr::select(-c(`subject#`, `total_UPDRS`, `motor_UPDRS`))
  
  md <- mahalanobis(X_mahal, colMeans(X_mahal), cov(X_mahal))
  df_clean$Mahalanobis_Dist <- md
  
  cutoff <- qchisq(p = 0.975, df = ncol(X_mahal))
  df_clean$Mahalanobis_Outlier <- md > cutoff
}

# Add 'is_outlier' column
X_flagged <- X_clean

if (!"Mahalanobis_Outlier" %in% colnames(df_clean)) {
  stop("❌ Mahalanobis_Outlier not found. Check STEP 3.7.")
}

X_flagged$is_outlier <- as.numeric(df_clean$Mahalanobis_Outlier)
cat("✅ Column 'is_outlier' added to X_flagged.\n")

# Remove extreme outliers (top 0.5%)
extreme_cutoff <- quantile(df_clean$Mahalanobis_Dist, 0.995)
non_extreme_idx <- which(df_clean$Mahalanobis_Dist <= extreme_cutoff)

X_clean <- X_flagged[non_extreme_idx, ]
y_total_clean <- y_total[non_extreme_idx]
y_motor_clean <- y_motor[non_extreme_idx]

# Final report
n_removed <- length(y_total) - length(y_total_clean)
cat("✅ Cleaned dataset from extreme outliers (Mahalanobis > 99.5th percentile):\n")
cat("   ➤ Rows removed:", n_removed, "\n")
cat("   ➤ Final shape: ", nrow(X_clean), "rows ×", ncol(X_clean), "columns\n")

# Optional: save filtered datasets
# write.csv(X_flagged, "X_flagged.csv", row.names = FALSE)
# write.csv(X_clean, "X_clean_filtered.csv", row.names = FALSE)
# write.csv(data.frame(y_total_clean = y_total_clean), "y_total_clean.csv", row.names = FALSE)
# write.csv(data.frame(y_motor_clean = y_motor_clean), "y_motor_clean.csv", row.names = FALSE)

# ===============================================================
# STEP 4.4 – Scaling & Normalization (Z-score Standardization)
#
# GOAL:
# • Standardize numeric features to mean = 0 and std = 1
# • Exclude the 'is_outlier' flag from the scaling process
# ===============================================================

# ---------------------------------------------------------------
# 1. Isolate numeric features (excluding 'is_outlier')
# ---------------------------------------------------------------
X_numeric <- X_clean[, sapply(X_clean, is.numeric)]
X_numeric <- X_numeric[, colnames(X_numeric) != "is_outlier"]

# ---------------------------------------------------------------
# 2. Z-score standardization (mean = 0, sd = 1)
# ---------------------------------------------------------------
X_scaled_matrix <- scale(X_numeric)

# ---------------------------------------------------------------
# 3. Convert to dataframe + add 'is_outlier' column
# ---------------------------------------------------------------
X_scaled <- as.data.frame(X_scaled_matrix)
X_scaled$is_outlier <- X_clean$is_outlier

# ---------------------------------------------------------------
# 4. Final check: mean ≈ 0 and standard deviation ≈ 1
# ---------------------------------------------------------------
cat("✅ Scaling completed.\n")
cat("   ➤ Mean (expected ~0):\n")
print(round(colMeans(X_scaled[, -ncol(X_scaled)]), 3))

cat("\n   ➤ Standard deviation (expected ~1):\n")
print(round(apply(X_scaled[, -ncol(X_scaled)], 2, sd), 3))

# ---------------------------------------------------------------
# 5. Final structure check
# ---------------------------------------------------------------
cat("\n🔍 Structure of the scaled dataset:\n")
str(X_scaled)

# Calculate Mahalanobis distance if not already present
if (!"Mahalanobis_Dist" %in% colnames(df_clean)) {
  X_mahal <- df_clean %>%
    dplyr::select(where(is.numeric)) %>%
    dplyr::select(-c(`subject#`, `total_UPDRS`, `motor_UPDRS`))
  
  md <- mahalanobis(X_mahal, colMeans(X_mahal), cov(X_mahal))
  df_clean$Mahalanobis_Dist <- md
}

# Outlier threshold (e.g., 97.5th percentile)
cutoff <- qchisq(0.975, df = ncol(X_mahal))

# Filter non-outlier rows
df_non_outlier <- df_clean[df_clean$Mahalanobis_Dist <= cutoff, ]

# Standardize numeric features for boxplot
X_scaled_filtered <- df_non_outlier %>%
  dplyr::select(where(is.numeric)) %>%
  dplyr::select(-c(`subject#`, `total_UPDRS`, `motor_UPDRS`, Mahalanobis_Dist))

X_scaled_matrix <- scale(X_scaled_filtered)
X_scaled_plot <- as.data.frame(X_scaled_matrix)

# Convert to long format
X_scaled_long <- reshape2::melt(X_scaled_plot, variable.name = "Feature", value.name = "Zscore")

# Plot
ggplot(X_scaled_long, aes(x = Feature, y = Zscore)) +
  geom_boxplot(fill = "#4DA8DA", color = "black", outlier.color = "red", outlier.size = 1) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 9)) +
  labs(title = "Boxplot", x = "Feature", y = "Z-score")

# Save
ggsave("EDA_plot/boxplot_mahalanobis_filtered.png", width = 12, height = 6, dpi = 300)





# ===============================================================
# STEP 4.5 – Train/Test Split + Proper Standardization
#
# GOAL:
# • Split dataset into training and test sets (80/20)
# • Apply z-score standardization ONLY on training data
#   and then scale the test set using the same parameters
# • Avoid data leakage
# ===============================================================

# ---------------------------------------------------------------
# 1. Set seed and create 80/20 split
# ---------------------------------------------------------------
set.seed(42)

train_index <- sample(seq_len(nrow(X_clean)), size = 0.8 * nrow(X_clean))

# Split predictors
X_train <- X_clean[train_index, ]
X_test  <- X_clean[-train_index, ]

# Split targets
y_total_train <- y_total_clean[train_index]
y_total_test  <- y_total_clean[-train_index]

y_motor_train <- y_motor_clean[train_index]
y_motor_test  <- y_motor_clean[-train_index]

cat("✅ Dataset split completed:\n")
cat("   ➤ Train set:", nrow(X_train), "observations\n")
cat("   ➤ Test set :", nrow(X_test), "observations\n\n")

# ---------------------------------------------------------------
# 2. Extract numeric columns to scale (exclude is_outlier)
# ---------------------------------------------------------------
X_train_numeric <- X_train[, sapply(X_train, is.numeric)]
X_train_numeric <- X_train_numeric[, colnames(X_train_numeric) != "is_outlier"]

X_test_numeric <- X_test[, sapply(X_test, is.numeric)]
X_test_numeric <- X_test_numeric[, colnames(X_test_numeric) != "is_outlier"]

# ---------------------------------------------------------------
# 3. Calculate mean and std ONLY from training set
# ---------------------------------------------------------------
means <- apply(X_train_numeric, 2, mean)
sds   <- apply(X_train_numeric, 2, sd)

# ---------------------------------------------------------------
# 4. Apply z-score scaling using training parameters only
# ---------------------------------------------------------------
X_train_scaled <- as.data.frame(scale(X_train_numeric, center = means, scale = sds))
X_test_scaled  <- as.data.frame(scale(X_test_numeric,  center = means, scale = sds))

# ---------------------------------------------------------------
# 5. Re-integrate the 'is_outlier' column
# ---------------------------------------------------------------
X_train_scaled$is_outlier <- X_train$is_outlier
X_test_scaled$is_outlier  <- X_test$is_outlier

# ---------------------------------------------------------------
# 6. Final structure check
# ---------------------------------------------------------------
cat("✅ Scaling successfully applied using training parameters.\n")
cat("   ➤ Size X_train_scaled:", nrow(X_train_scaled), "×", ncol(X_train_scaled), "\n")
cat("   ➤ Size X_test_scaled :", nrow(X_test_scaled), "×", ncol(X_test_scaled), "\n")

# (Optional) Check mean and std after scaling
cat("\n🔍 Check: mean of training features (≈ 0):\n")
print(round(colMeans(X_train_scaled[, -ncol(X_train_scaled)]), 3))

cat("\n🔍 Check: std of training features (≈ 1):\n")
print(round(apply(X_train_scaled[, -ncol(X_train_scaled)], 2, sd), 3))


# ===============================================================
# STEP 4.6 – Prepare Structured Datasets for Modeling (Dual Target)
# ===============================================================

library(dplyr)

# ---------------------------------------
# 1. RAW version: with outliers, scaled
# ---------------------------------------
X_raw_numeric <- X_flagged %>%
  dplyr::select(where(is.numeric)) %>%
  dplyr::select(-is_outlier)

means_raw <- apply(X_raw_numeric, 2, mean)
sds_raw   <- apply(X_raw_numeric, 2, sd)

X_raw_scaled <- as.data.frame(scale(X_raw_numeric, center = means_raw, scale = sds_raw))
X_raw_scaled$is_outlier <- X_flagged$is_outlier

# Targets
y_total_raw <- y_total
y_motor_raw <- y_motor

cat("✅ X_raw_scaled and y_raw_* prepared (outliers included).\n")

# ---------------------------------------
# 2. CLEAN version: filtered, scaled
# ---------------------------------------
X_clean_numeric <- X_clean %>%
  dplyr::select(where(is.numeric)) %>%
  dplyr::select(-is_outlier)

means_clean <- apply(X_clean_numeric, 2, mean)
sds_clean   <- apply(X_clean_numeric, 2, sd)

X_clean_scaled <- as.data.frame(scale(X_clean_numeric, center = means_clean, scale = sds_clean))
X_clean_scaled$is_outlier <- X_clean$is_outlier

cat("✅ X_clean_scaled and y_clean_* prepared (no extreme outliers).\n")

# ---------------------------------------
# 3. PCA version (optional, on X_clean_scaled)
# ---------------------------------------
pca_result <- prcomp(X_clean_scaled %>% dplyr::select(-is_outlier), center = FALSE, scale. = FALSE)

explained_var <- summary(pca_result)$importance["Cumulative Proportion", ]
n_components <- which(explained_var >= 0.95)[1]

X_pca <- as.data.frame(pca_result$x[, 1:n_components])

# Targets
y_total_pca <- y_total_clean
y_motor_pca <- y_motor_clean

explained_variance_vector <- summary(pca_result)$importance[2, 1:n_components]

cat("✅ X_pca prepared with", n_components, "components (≥95% variance).\n")

# ===============================================================
# Save everything
# ===============================================================

output_dir <- "Processed"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
  cat("📁 Folder 'Processed' created.\n")
}

# --- Save predictor datasets ---
write.csv(X_raw_scaled,   file = file.path(output_dir, "X_raw_scaled.csv"),   row.names = FALSE)
write.csv(X_clean_scaled, file = file.path(output_dir, "X_clean_scaled.csv"), row.names = FALSE)
write.csv(X_pca,          file = file.path(output_dir, "X_pca.csv"),          row.names = FALSE)

# --- Save target variables ---
write.csv(data.frame(y_total_raw = y_total_raw), file = file.path(output_dir, "y_total_raw.csv"), row.names = FALSE)
write.csv(data.frame(y_motor_raw = y_motor_raw), file = file.path(output_dir, "y_motor_raw.csv"), row.names = FALSE)

write.csv(data.frame(y_total_clean = y_total_clean), file = file.path(output_dir, "y_total_clean.csv"), row.names = FALSE)
write.csv(data.frame(y_motor_clean = y_motor_clean), file = file.path(output_dir, "y_motor_clean.csv"), row.names = FALSE)

write.csv(data.frame(y_total_pca = y_total_pca), file = file.path(output_dir, "y_total_pca.csv"), row.names = FALSE)
write.csv(data.frame(y_motor_pca = y_motor_pca), file = file.path(output_dir, "y_motor_pca.csv"), row.names = FALSE)

# --- Save explained variance of PCA ---
write.csv(data.frame(PC = paste0("PC", 1:length(explained_variance_vector)),
                     VarianceExplained = explained_variance_vector),
          file = file.path(output_dir, "pca_variance.csv"),
          row.names = FALSE)

cat("✅ All datasets saved to 'Processed' folder.\n")

# --- Logging ---
log_file <- file.path(output_dir, "processing_log.txt")
log_lines <- c(
  paste0("🕒 Timestamp: ", Sys.time()),
  "✅ X_raw_scaled: all samples, scaled, with outliers",
  "✅ X_clean_scaled: Mahalanobis-filtered, scaled",
  paste0("✅ X_pca: ", ncol(X_pca), " components (≥95% variance)"),
  "✅ y_total_* and y_motor_* saved",
  "✅ PCA variance saved",
  "------------------------------------------ --------\n"
)
writeLines(log_lines, con = log_file, sep = "\n", append = TRUE)
cat("📝 Log updated at:", log_file, "\n")
