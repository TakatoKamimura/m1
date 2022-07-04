import pytchat
import time
import openpyxl
wb = openpyxl.Workbook()
sheet=wb.active
sheet.title="チャットリプレイ"

chat = pytchat.create(video_id="35Sf6u13oKM")
cnt=0
cnt1=1
end_time=0
start_time=0
while chat.is_alive():
    chatdata=chat.get().sync_items()
    for c in chatdata:
        if cnt>0:
            end_time = time.time()
        # start_time = time.time()
        print(f"{c.timestamp},{c.message}")
        if cnt>0:
            print(end_time-start_time)
        sel1="A"+str(cnt1)
        sel2="B"+str(cnt1)
        sheet[sel1].value=c.timestamp
        sheet[sel2].value=c.message
        cnt1+=1
        start_time = time.time()
        # print(end_time-start_time)
chat.terminate()
wb.save("text2.csv")
wb.close()