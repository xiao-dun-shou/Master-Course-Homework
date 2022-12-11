from scrapy_redis.spiders import RedisSpider
from test1.items import Test1Item

class MyspiderSpider(RedisSpider):
    name = 'myspider'   #爬虫名
    allowed_domains = ['www.ly.com']

    def parse(self, response,**kwargs):
        text = response.text
        try:
            for j in range(1, 11):
                comment = text.split('dpContent":"')[j].split('","dpDate')[0]
                dp = text.split('"lineAccess":"')[j].split('","markColor"')[0]
                if dp == "好评":
                    dp = 1
                else:
                    dp = 0
                yield Test1Item(comment=comment,score=dp)
        except:
            print("*"*20 +"empty html, skip" +"*"*20)
            pass