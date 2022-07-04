import requests
id='b0mRw0AyCbc'
apikey='AIzaSyBrC3uscXDqnBFUmh1fEqBis7_AwZV2LeA'
url = 'https://www.googleapis.com/youtube/v3/videos?id='+id+'&key='+apikey+'&part=snippet,contentDetails,statistics,status'

response = requests.get(url)
print(response.json())
print("https://www.youtube.com/watch?v="+response.json()['items'][0]['id'])
print(response.json()['items'][0]['snippet']['description'])