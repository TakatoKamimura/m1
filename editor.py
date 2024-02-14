from moviepy.editor import *

class editor():
    def __init__(self,start,end):
        self.start=start
        self.end=end
    
    def cut(start,end):
        return VideoFileClip("動画ファイルのパス").subclip(start,end)

    def concatenate(clip:list):
        return concatenate_videoclips(clip)
    
    def save(a):
        a.write_videofile('出力したいファイルのパス')


