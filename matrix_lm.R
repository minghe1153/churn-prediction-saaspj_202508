getwd()
setwd("/Users/minghe328/playground")
file.exists("dataset_ica_ml_train.csv")
# まずはデータを読み込んで構造確認
df <- read.csv("dataset_ica_ml_train.csv")
str(df)
summary(df)

# 最初は基本的なpairs()から開始
pairs(df[c("Age", "Balance", "EstimatedSalary")])

# GGallyで詳細分析
install.packages("GGally", dependencies = TRUE)
library(GGally)
ggpairs(df, columns = c("Age", "Balance", "EstimatedSalary", "Exited"))

# 「Balance」の確認
df_categorized <- df %>%
  mutate(
    BalanceCategory = case_when(
      Balance == 0 ~ "Zero",
      Balance > 0 & Balance <= 50000 ~ "1-50K",
      Balance > 50000 & Balance <= 100000 ~ "50K-100K",
      Balance > 100000 & Balance <= 150000 ~ "100K-150K",
      Balance > 150000 ~ "150K+"
    ),
    BalanceCategory = factor(BalanceCategory, 
                             levels = c("Zero", "1-50K", "50K-100K", "100K-150K", "150K+"))
  )

# 分布確認
table(df_categorized$BalanceCategory)

# 解約率の区間別分析
churn_by_balance <- tapply(df_categorized$Exited, df_categorized$BalanceCategory, mean)
print(churn_by_balance)

# より詳細な分析
df_categorized %>%
  group_by(BalanceCategory) %>%
  summarise(
    count = n(),
    churn_rate = mean(Exited),
    avg_age = mean(Age),
    avg_salary = mean(EstimatedSalary)
  )

# 年齢区間別の解約率
age_analysis <- df %>%
  mutate(AgeGroup = cut(Age, breaks = c(0, 30, 40, 50, 60, Inf))) %>%
  group_by(AgeGroup) %>%
  summarise(count = n(), churn_rate = mean(Exited))

print(age_analysis)

# 年齢×残高カテゴリの相互作用
age_balance_interaction <- df_categorized %>%
  mutate(AgeGroup = cut(Age, breaks = c(0, 30, 40, 50, 60, Inf))) %>%
  group_by(AgeGroup, BalanceCategory) %>%
  summarise(count = n(), churn_rate = mean(Exited), .groups = 'drop') %>%
  arrange(AgeGroup, BalanceCategory)

print(age_balance_interaction, n = Inf)

# 国別の基本解約率確認
df %>%
  group_by(Geography) %>%
  summarise(count = n(), churn_rate = mean(Exited))

# 国×年齢の相互作用
df %>%
  mutate(AgeGroup = cut(Age, breaks = c(0, 40, 50, 60, Inf))) %>%
  group_by(Geography, AgeGroup) %>%
  summarise(count = n(), churn_rate = mean(Exited), .groups = 'drop')

# 国×残高カテゴリの相互作用
df_categorized %>%
  group_by(Geography, BalanceCategory) %>%
  summarise(count = n(), churn_rate = mean(Exited), .groups = 'drop') %>%
  arrange(Geography, BalanceCategory)

# ドイツの残高分布を確認
table(df$Geography, df$Balance == 0)

# 年齢の二重経路の確認
cor(df$Age, df$Balance)  # 間接効果

# 統計モデル
model <- glm(Exited ~ Geography + Age + BalanceCategory, 
             data = df_categorized, family = binomial)
summary(model)

# 予測確率の計算
predictions_prob <- predict(model, type = "response")

# 予測ラベル（閾値0.5）
predictions_class <- ifelse(predictions_prob > 0.5, 1, 0)

# 基本的な混同行列
conf_matrix <- table(Predicted = predictions_class, Actual = df_categorized$Exited)
print(conf_matrix)

# 手動で評価指標を計算
TP <- conf_matrix[2,2]  # True Positive
TN <- conf_matrix[1,1]  # True Negative  
FP <- conf_matrix[2,1]  # False Positive
FN <- conf_matrix[1,2]  # False Negative

# 評価指標
accuracy <- (TP + TN) / sum(conf_matrix)
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1 <- 2 * (precision * recall) / (precision + recall)

cat("Accuracy:", round(accuracy, 3), "\n")
cat("Precision:", round(precision, 3), "\n")
cat("Recall:", round(recall, 3), "\n")
cat("F1-Score:", round(f1, 3), "\n")

# 複数の閾値でTPRとFPRを計算
thresholds <- seq(0, 1, by = 0.01)
tpr_values <- numeric(length(thresholds))
fpr_values <- numeric(length(thresholds))

for (i in seq_along(thresholds)) {
  pred_class <- ifelse(predictions_prob >= thresholds[i], 1, 0)
  conf_mat <- table(Predicted = pred_class, Actual = df_categorized$Exited)
  
  if (nrow(conf_mat) == 2 && ncol(conf_mat) == 2) {
    TP <- conf_mat[2,2]
    FN <- conf_mat[1,2] 
    FP <- conf_mat[2,1]
    TN <- conf_mat[1,1]
  } else {
    # 極端な閾値での処理
    if (all(pred_class == 0)) {
      TP <- 0; FN <- sum(df_categorized$Exited == 1)
      FP <- 0; TN <- sum(df_categorized$Exited == 0)
    } else {
      TP <- sum(df_categorized$Exited == 1); FN <- 0
      FP <- sum(df_categorized$Exited == 0); TN <- 0
    }
  }
  
  tpr_values[i] <- TP / (TP + FN)  # 真陽性率
  fpr_values[i] <- FP / (FP + TN)  # 偽陽性率
}

# AUC計算（台形則）
auc_value <- sum(diff(fpr_values) * (tpr_values[-1] + tpr_values[-length(tpr_values)]) / 2)

cat("ROC-AUC:", round(abs(auc_value), 3), "\n")

# ROC曲線描画
plot(fpr_values, tpr_values, type = "l", 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     main = paste("ROC Curve (AUC =", round(abs(auc_value), 3), ")"))
abline(0, 1, lty = 2, col = "red")  # ランダム予測線

# 最適な閾値を探す
thresholds <- c(0.3, 0.25, 0.2, 0.15)

for (thresh in thresholds) {
  pred_class <- ifelse(predictions_prob > thresh, 1, 0)
  conf_mat <- table(Predicted = pred_class, Actual = df_categorized$Exited)
  
  if (nrow(conf_mat) == 2 && ncol(conf_mat) == 2) {
    recall <- conf_mat[2,2] / sum(conf_mat[,2])
    precision <- conf_mat[2,2] / sum(conf_mat[2,])
    f1 <- 2 * (precision * recall) / (precision + recall)
    
    cat("閾値", thresh, ": Recall =", round(recall, 3), 
        ", Precision =", round(precision, 3), 
        ", F1 =", round(f1, 3), "\n")
  }
}

# 閾値0.2での最終評価
optimal_threshold <- 0.2
pred_optimal <- ifelse(predictions_prob > optimal_threshold, 1, 0)
conf_optimal <- table(Predicted = pred_optimal, Actual = df_categorized$Exited)
print(conf_optimal)

cat("最適閾値0.2での性能:\n")
cat("- 解約者の68%を正しく予測\n")
cat("- 予測した解約者の38.2%が実際に解約\n")
