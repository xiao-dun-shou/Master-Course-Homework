import redis
import requests
import time
import tqdm

pool = redis.ConnectionPool(host="127.0.0.1", port=6379)
redisInstace = redis.Redis(connection_pool=pool)
redisInstace.flushdb()
spider_name = "myspider"

sleeptime = 1
regionMax = 1

#获取景点id
sid_sess = requests.session()
regionidList = range(0,regionMax)
sidlist = []
for regionid in regionidList:
    for page in tqdm.tqdm(range(1,3)):
        url = 'https://www.ly.com/scenery/scenerysearchlist_32_394__0_'+str(regionid)+'_0_0_0_0_100'+str(page)+'.html'
        htmlContent = sid_sess.get(url).text
        time.sleep(sleeptime)
    for i in range(1,999):
            try:
                sidlist.append(htmlContent.split('href="/scenery/BookSceneryTicket_')[i].split('.html')[0])
            except:
                break

#根据景区id获取url存储到redis中
for sid in tqdm.tqdm(sidlist):
    for i in tqdm.tqdm(range(0,99)):
        url = 'https://www.ly.com/scenery/AjaxHelper/DianPingAjax.aspx?action=GetDianPingList&sid='+sid+'&page='+str(i)+'&pageSize=10&labId=1&sort=0&iid=0.6401678021327853'
        redisInstace.rpush(spider_name+":start_urls", url)
print(redisInstace.lrange(spider_name+":start_urls",0,-1))


