import redis
import json
import xlwt

savepath = "./results.xls"

pool = redis.ConnectionPool(host="127.0.0.1", port=6379)
redisInstace = redis.Redis(connection_pool=pool)
spider_name = "myspider"

comments = []
scores = []
for data_json in redisInstace.lrange(spider_name+":items",0,-1):
    data = json.loads(data_json.decode("utf-8"))
    comment = data["comment"]
    score = data["score"]
    comments.append(comment)
    scores.append(score)

workbook = xlwt.Workbook(encoding="utf-8")
worksheet = workbook.add_sheet('sheet1')
worksheet.write(0,0,label='label')
worksheet.write(0,1,label='review')
for i in range(len(comments)):
    worksheet.write(i+1,0,label=scores[i])
    worksheet.write(i+1,1,label=comments[i])
workbook.save(savepath)