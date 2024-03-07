from moviepy.editor import *

class editor():
    def __init__(self,start,end):
        self.start=start
        self.end=end
    # 動画ファイル内のstart秒からend秒までをクリップしてリターン
    def cut(start,end):
        return VideoFileClip("動画ファイルのパス").subclip(start,end)
    # 複数動画を一つの動画に統合
    def concatenate(clip:list):
        return concatenate_videoclips(clip)
    # 生成したハイライト動画の保存
    def save(a):
        a.write_videofile('出力したいファイルのパス')


