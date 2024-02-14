import pytchat
import time
import openpyxl as op
import re

wb = op.Workbook()
sheet=wb.active
sheet.title="チャットリプレイ"
stream = pytchat.create(video_id = "テキストチャットを取得したいアーカイブ動画のURLの'?v='の後に続いている文字列")
cnt=2
cnt1=0
i = 0
sheet["A1"].value=('時間')
sheet["B1"].value=('コメント')
start_time=time.time()
while stream.is_alive():
  data = stream.get()
  items = data.items
  for c in items:
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    if re.search(pattern,c.message):
      continue
    result = re.sub(':.*?:', '', c.message)
    if len(result)<1:
      continue
    print(f"{c.timestamp},{result}")
    sel1="A"+str(cnt)
    sel2="B"+str(cnt)
    if cnt1==0:
        start=c.timestamp
        cnt1+=1
    sheet[sel1].value=(c.timestamp-start)/1000
    sheet[sel2].value=result
    cnt+=1
  time.sleep(3)
  i += 1
stream.terminate
end_time=time.time()
print(end_time-start_time)
stream.terminate()
wb.save("出力先のパス")
wb.close()