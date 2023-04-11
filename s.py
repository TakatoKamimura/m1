import re
# s = 'this is :sample string: for _extracting substring_.'

# # アスタリスクで囲まれている部分を抽出
# p = r':(.*):'  # アスタリスクに囲まれている任意の文字
# #p = r'\*[^*]*\*'  # アスタリスクに囲まれているアスタリスク以外の文字
# r = re.findall(p, s)  # パターンに当てはまるものを全て抽出
# print(r)
s=':face_with_tears_of_joy:'
result = re.sub(':.*?:', '', s)  # : で囲まれた部分を除去
print(result)
print(len(result))
# p=r':(.*):'
# r=re.findall(p,s)
# print(r)