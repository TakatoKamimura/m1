from yt_dlp import YoutubeDL

# ダウンロード条件を設定する。今回は画質・音質ともに最高の動画をダウンロードする
ydl_opts = {'format': 'best',
            'outtmpl': 'movie\%(title)s.%(ext)s'}

# 動画のURLを指定
with YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=lYJE1CBf_2o'])