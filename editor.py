from moviepy.editor import *

class editor():
    def __init__(self,start,end):
        self.start=start
        self.end=end
    
    def cut(start,end):
        return VideoFileClip("movie\\hinano.mp4").subclip(start,end)

    def concatenate(clip:list):
        return concatenate_videoclips(clip)
    
    def save(a):
        a.write_videofile('movie\\output.mp4')


