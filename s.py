import re
from googletrans import Translator
# s = 'this is :sample string: for _extracting substring_.'

# # アスタリスクで囲まれている部分を抽出
# p = r':(.*):'  # アスタリスクに囲まれている任意の文字
# #p = r'\*[^*]*\*'  # アスタリスクに囲まれているアスタリスク以外の文字
# r = re.findall(p, s)  # パターンに当てはまるものを全て抽出
# print(r)
s=':kuzuha::nice::nice:'
s=list(map(str,s.split(':')))
a=[]
for i in range(len(s)):
    if len(s[i])>0:
        a.append(s[i])
traslator=Translator()
for i in range(len(a)):
    t=traslator.translate(a[i],src='en',dest="ja")
    print(t.text)

# p=r':(.*):'
# r=re.findall(p,s)
# print(r)