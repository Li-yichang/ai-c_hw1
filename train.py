from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import normalized_mutual_info_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# 讀取標註好的留言 CSV
df = pd.read_csv("youtube_comments_with_sentiment.csv")

# 留言和情緒
X = df["Comment"]
y = df["Sentiment"]

# TF-IDF 向量化
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# 解決類別不均（SMOTE 只對數值特徵適用）
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# 八成訓練，兩成測試
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ========== 🚀 監督學習（Supervised Learning） ==========

## Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

print("🔹 Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("🔹 Naive Bayes Report:\n", classification_report(y_test, y_pred_nb, zero_division=0))

## SVM（加權，處理不均衡數據）
svm_model = SVC(kernel='linear', class_weight='balanced')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("🔹 SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("🔹 SVM Report:\n", classification_report(y_test, y_pred_svm, zero_division=0))

# ========== 🎯 交叉驗證（Cross Validation） ==========
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nb_cv_score = cross_val_score(nb_model, X_resampled, y_resampled, cv=kf, scoring='accuracy').mean()
svm_cv_score = cross_val_score(svm_model, X_resampled, y_resampled, cv=kf, scoring='accuracy').mean()

print(f"📊 Naive Bayes CV Accuracy: {nb_cv_score:.4f}")
print(f"📊 SVM CV Accuracy: {svm_cv_score:.4f}")

# ========== 🏷️ 無監督學習（Unsupervised Learning） ==========

## K-means 聚類
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(X_tfidf)
kmeans_labels = kmeans.labels_

# 計算 K-means 聚類與真實標籤的相似度
nmi_score = normalized_mutual_info_score(y.astype(str), kmeans_labels)
print(f"🔹 K-means NMI Score: {nmi_score:.4f}")

# 顯示部分聚類結果
cluster_df = pd.DataFrame({"Comment": X, "Cluster": kmeans_labels, "Sentiment": y})
print("\n🔍 K-means Clustering Results:")
print(cluster_df.head(10))
