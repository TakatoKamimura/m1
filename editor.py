from moviepy.editor import *

class editor():
    def __init__(self,start,end):
        self.start=start
        self.end=end
    
    def cut(start,end):
        return VideoFileClip("movie\\kuzuha_vcc.mp4").subclip(start,end)

    def concatenate(clip:list):
        return concatenate_videoclips(clip)
    
    def save(a):
        a.write_videofile('movie\\lYJE1CBf_2o_28kuzuha_kirinukich_Wrime無し_batch8_val改善_output.mp4')


