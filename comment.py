import os
import csv
from googleapiclient.discovery import build

# 設定 YouTube API 金鑰
API_KEY = "AIzaSyDv_xqmuOW8jtSEIblUWurwoPiybqFP5Qg"  
youtube = build("youtube", "v3", developerKey=API_KEY)

# 設定要抓取的影片 ID（可以從 YouTube 影片網址取得）
VIDEO_ID = "1mvWQObfyHk"  # 例如 https://www.youtube.com/watch?v=dQw4w9WgXcQ，影片ID就是 dQw4w9WgXcQ

# 取得影片留言
def get_youtube_comments(video_id, max_results=500):
    comments = []
    
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=max_results
    )
    
    response = request.execute()
    
    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)
    
    return comments

# 取得留言並存入 CSV
comments = get_youtube_comments(VIDEO_ID)

# 存成 CSV
csv_filename = "youtube_comments.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Comment"])
    for comment in comments:
        writer.writerow([comment])

print(f"留言已儲存至 {csv_filename}")
