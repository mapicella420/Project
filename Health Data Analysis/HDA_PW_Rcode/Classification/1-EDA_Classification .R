# =====================================================
# REQUIRED PACKAGES INSTALLATION (RUN ONLY ONCE)
# =====================================================
install.packages("ggfortify")
install.packages("dplyr")
install.packages("tidyverse")

# =====================================================
# CLASSIFICATION PIPELINE – INITIAL EXPLORATION
# =====================================================

# ==========================
# Load required libraries
# ==========================
library(readr)
library(dplyr)
library(ggplot2)
library(corrplot)
library(reshape2)
library(scales)
library(rlang)
library(car)
library(stats)
library(tidyverse)
library(tidyr)
library(ggfortify)
library(caret)
library(glmnet)
library(pROC)
library(tibble)
library(e1071)

# ==========================
# Load and clean dataset
# ==========================
setwd("/Users/salvatore/Desktop/HealtDataAnalytics/Classification")
df <- read_csv("Prostate.csv")

# Remove uninformative columns
df <- df %>% select(-matches("^samples$|^X$|^index$", ignore.case = TRUE))

# Convert response to factor and clean labels
df$response <- as.factor(df$response)
df$target <- df$response
df$response <- NULL
levels(df$target)[levels(df$target) == "tumer"] <- "tumor"

# ==========================
# Dataset overview
# ==========================
cat("Dataset dimensions: ", dim(df)[1], "rows ×", dim(df)[2], "columns\n")
head(df)
str(df)
summary(df)
table(df$target)
prop.table(table(df$target))

# Check for missing values
missing_counts <- colSums(is.na(df))
cat("Total columns with missing values:", sum(missing_counts > 0), "\n")

# ==========================
# T-test-based feature selection
# ==========================
features <- df[, grepl("^V", colnames(df))]
target <- df$target

p_values <- apply(features, 2, function(x) t.test(x ~ target)$p.value)
top_features <- names(sort(p_values))[1:6]

# ==========================
# Density plots for top 5 features
# ==========================
if (!dir.exists("EDA_plot")) dir.create("EDA_plot")

for (feature_name in top_features) {
  plot_df <- df %>%
    select(all_of(c(feature_name, "target"))) %>%
    rename(Value = all_of(feature_name), Class = target)
  
  p <- ggplot(plot_df, aes(x = Value, fill = Class, color = Class)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, alpha = 0.25, position = "identity") +
    geom_density(alpha = 0.4, linewidth = 1.2) +
    scale_fill_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
    scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
    theme_minimal(base_size = 14) +
    labs(
      title = paste("Distribution of", feature_name, "by Class"),
      x = paste("Expression Level of", feature_name),
      y = "Density",
      fill = "Class",
      color = "Class"
    )
  
  ggsave(filename = paste0("EDA_plot/density_", feature_name, ".png"),
         plot = p, width = 7, height = 5, dpi = 300)
  print(p)
}

# ==========================
# Summary statistics for top 5 by variance
# ==========================
top_features_var <- names(sort(apply(df[, -ncol(df)], 2, var), decreasing = TRUE))[1:5]

summary_table <- df %>%
  select(all_of(top_features_var), target) %>%
  group_by(target) %>%
  summarise(across(everything(),
                   list(mean = mean, sd = sd, median = median, min = min, max = max),
                   .names = "{.col}_{.fn}"),
            .groups = "drop")

print(summary_table)

# ==========================
# T-test summary for selected features
# ==========================
ttest_results <- p_values[top_features_var]
print(sort(ttest_results))
# =============================================
# Boxplot with annotated quartiles and extrema
# =============================================

# Calculate p-values and select top 6 features
p_values <- apply(df[, -which(names(df) == "target")], 2, function(col) {
  t.test(col ~ df$target)$p.value
})
top_features <- names(sort(p_values))[1:6]

# Loop for each top feature
for (feature in top_features) {
  
  # Compute summary stats per class
  stats_df <- df %>%
    group_by(target) %>%
    summarise(
      Min = min(.data[[feature]]),
      Q1 = quantile(.data[[feature]], 0.25),
      Median = median(.data[[feature]]),
      Q3 = quantile(.data[[feature]], 0.75),
      Max = max(.data[[feature]]),
      .groups = "drop"
    )
  
  # Base boxplot
  p <- ggplot(df, aes(x = target, y = .data[[feature]], fill = target)) +
    geom_boxplot(alpha = 0.6, outlier.shape = NA) +
    geom_jitter(width = 0.2, alpha = 0.2) +
    
    # Annotate stats: positioned to the right (hjust < 0), spaced by y
    geom_text(data = stats_df, aes(x = target, y = Min, label = "Min"), 
              hjust = -0.6, vjust = 0.5, size = 3.5, color = "black") +
    geom_text(data = stats_df, aes(x = target, y = Q1, label = "1st Quartile"), 
              hjust = -0.6, vjust = 0.5, size = 3.5, color = "black") +
    geom_text(data = stats_df, aes(x = target, y = Median, label = "Median"), 
              hjust = -0.6, vjust = 0.5, size = 3.5, color = "black", fontface = "bold") +
    geom_text(data = stats_df, aes(x = target, y = Q3, label = "3rd Quartile"), 
              hjust = -0.6, vjust = 0.5, size = 3.5, color = "black") +
    geom_text(data = stats_df, aes(x = target, y = Max, label = "Max"), 
              hjust = -0.6, vjust = 0.5, size = 3.5, color = "black") +
    
    # Labels and theme
    labs(
      title = paste0("Boxplot of ", feature, " by Class"),
      x = "Class (Normal vs Tumor)",
      y = paste("Expression Level of", feature)
    ) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none") +
    scale_fill_manual(values = c("normal" = "#0072B2", "tumor" = "#D55E00")) +
    coord_cartesian(clip = "off")  # Allows text to go outside panel
  
  # Print and save
  print(p)
  ggsave(filename = paste0("EDA_plot/boxplot_annotated_", feature, ".png"),
         plot = p, width = 7, height = 5, dpi = 300)
}


# ===============================================================
# Principal Component Analysis (PCA) – Top 30 Most Variable Features
# ===============================================================


# 1. Estrai le feature numeriche
features <- df %>% select(starts_with("V"))
features_numeric <- features[, sapply(features, is.numeric)]

# 2. Calcola la varianza
variances <- apply(features_numeric, 2, var, na.rm = TRUE)
top_features <- names(sort(variances, decreasing = TRUE))[1:30]

# 3. Subset e standardizzazione
df_top <- df[, top_features]
df_top_scaled <- scale(df_top)

# 4. PCA
pca_res <- prcomp(df_top_scaled)

# 5. Varianza spiegata
explained <- summary(pca_res)$importance[2, 1:2] * 100
x_lab <- paste0("PC1 (", round(explained[1], 1), "%)")
y_lab <- paste0("PC2 (", round(explained[2], 1), "%)")

# 6. Prepara il target
target <- as.factor(df$target)
levels(target) <- c("normal", "tumor")

# 7. PCA plot
pca_plot <- autoplot(
  pca_res,
  data = data.frame(Class = target),
  colour = "Class",
  shape = "Class",
  loadings = FALSE
) +
  scale_color_manual(values = c("normal" = "#1f77b4", "tumor" = "#d62728")) +
  scale_shape_manual(values = c("normal" = 16, "tumor" = 4)) +
  labs(
    title = "PCA – Top 30 Most Variable Features",
    x = x_lab,
    y = y_lab
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")

# Salva e mostra
if (!dir.exists("EDA_plot")) dir.create("EDA_plot")
ggsave("EDA_plot/pca_plot.png", pca_plot, width = 8, height = 6, dpi = 300)
print(pca_plot)  # Mostra anche nella sezione Plots

# 8. PCA Loadings plot – PC1
loadings_df <- as.data.frame(pca_res$rotation[, 1])
colnames(loadings_df) <- "Loading_PC1"
loadings_df$Feature <- rownames(loadings_df)

loadings_df <- loadings_df %>%
  arrange(desc(abs(Loading_PC1))) %>%
  head(30)

loadings_plot <- ggplot(loadings_df, aes(x = reorder(Feature, Loading_PC1), y = Loading_PC1)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal(base_size = 13) +
  labs(
    title = "Top 30 Feature Loadings on PC1",
    x = "Feature",
    y = "Loading Weight"
  )

ggsave("EDA_plot/pca_loadings_PC1.png", loadings_plot, width = 8, height = 6, dpi = 300)
print(loadings_plot)  # Mostra anche nella sezione Plots

# Mostra varianza spiegata
summary(pca_res)$importance[2, 1:2] * 100
