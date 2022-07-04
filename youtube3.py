import pytchat
import time
import openpyxl as op
wb = op.Workbook()
sheet=wb.active
sheet.title="チャットリプレイ"
stream = pytchat.create(video_id = "b0mRw0AyCbc")
cnt=2
cnt1=0
i = 0
start_time=time.time()
while stream.is_alive():
  data = stream.get()
  items = data.items
  for c in items:
      print(f"{c.timestamp},{c.message}")
      sel1="A"+str(cnt)
      sel2="B"+str(cnt)
      if cnt1==0:
          start=c.timestamp
          cnt1+=1
      sheet[sel1].value=(c.timestamp-start)/1000
      sheet[sel2].value=c.message
      cnt+=1
  time.sleep(3)
  i += 1
  #if i == 10:
    # get the continuation parameter
   # continuation = stream.continuation
    #stream.terminate()
    #break
stream.terminate
end_time=time.time()
print(end_time-start_time)
stream.terminate()
wb.save("ひなーの.csv")
wb.close()

# retrieve chatdata from the continuation.
#stream = pytchat.create(video_id = "35Sf6u13oKM", replay_continuation=continuation)
#data = stream.get()
#items = data.items
#for c in items:
#    print(f"{c.datetime} [{c.author.name}]- {c.message}")
#stream.terminate()