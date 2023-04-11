from moviepy.editor import *

class editor():
    def __init__(self,start,end):
        self.start=start
        self.end=end
    
    def cut(self):
        return VideoFileClip("movie\\ニシダの小説書.mp4").subclip(self.start,self.end)

    def concatenate(clip:list):
        return concatenate_videoclips(clip)

