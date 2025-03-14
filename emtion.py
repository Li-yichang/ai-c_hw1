import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

#  讀留言
df = pd.read_csv("youtube_comments.csv")

sia = SentimentIntensityAnalyzer()

#本來下面想靠套件做情緒分系，但是還是不太靈光所以後來還是有人工改動
# 進行情緒分析
def analyze_sentiment(comment):
    score = sia.polarity_scores(comment)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# 新增情緒標籤
df["Sentiment"] = df["Comment"].apply(analyze_sentiment)

# 存成新的 CSV
df.to_csv("youtube_comments_with_sentiment.csv", index=False, encoding="utf-8")
print("已完成情緒標註！")
