import pandas as pd

# CSVファイルからデータをUTF-8で読み込む（実際のファイルパスに適応してください）
input_file_path = 'textchat_from_youtube\\JmOGWt-XjzI(葛葉切り抜き集用).csv'
df = pd.read_csv(input_file_path, encoding='utf-8')

# 特定のキーワードが含まれるコメントのラベルを1に変更
keywords = ['w', 'W', 'ｗ', 'Ｗ', '草', '笑','ないす','ナイス','惜しい','おっし','おしい','おしぃ']
condition = df['コメント'].str.contains('|'.join(keywords), case=False)
df.loc[condition, 'ラベル'] = 1
# 新しいCSVファイルにUTF-8で保存する（実際のファイルパスに適応してください）
output_file_path = 'JmOGWt-XjzI(葛葉切り抜き集用)_wを1に.csv'
df.to_csv(output_file_path, index=False, encoding='utf-8')
