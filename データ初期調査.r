# 現在のディレクトリ確認
print(paste("現在のディレクトリ:", getwd()))

# デスクトップに移動
setwd("~/Desktop")

# 移動後のディレクトリ確認
print(paste("移動後のディレクトリ:", getwd()))

# デスクトップのファイル一覧確認
print("デスクトップのファイル一覧:")
print(list.files())

# readrパッケージを読み込み
library(readr)

# datasetフォルダ内のCSVファイルを読み込み
data_train <- read_csv("dataset/dataset_ica_ml_train.csv")
data_test <- read_csv("dataset/dataset_ica_ml_test.csv")

# データの確認
print("trainデータの構造:")
str(data_train)
print("trainデータの最初の数行:")
head(data_train)

# 基本統計量の表示（明示的にprintを使用）
print("trainデータの基本統計量:")
print(summary(data_train))

# 欠損値の確認
print("欠損値の数:")
print(colSums(is.na(data_train)))

# 欠損値の割合
print("欠損値の割合(%):")
print(round(colSums(is.na(data_train)) / nrow(data_train) * 100, 2))

# データの形状
print(paste("データの形状:", nrow(data_train), "行", ncol(data_train), "列"))

# 数値列のみの詳細統計量
numeric_cols <- sapply(data_train, is.numeric)
if(any(numeric_cols)) {
  print("数値列の詳細統計量:")
  print(summary(data_train[, numeric_cols]))
  
  # 外れ値の確認（四分位範囲法）
  print("外れ値の可能性がある値の数:")
  for(col in names(data_train)[numeric_cols]) {
    Q1 <- quantile(data_train[[col]], 0.25, na.rm = TRUE)
    Q3 <- quantile(data_train[[col]], 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    outliers <- sum(data_train[[col]] < (Q1 - 1.5 * IQR) | 
                   data_train[[col]] > (Q3 + 1.5 * IQR), na.rm = TRUE)
    print(paste(col, ":", outliers, "個"))
  }
}

# データの形状
print(paste("データの形状:", nrow(data_train), "行", ncol(data_train), "列"))