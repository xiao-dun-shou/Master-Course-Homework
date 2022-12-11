BOT_NAME = 'test1'

SPIDER_MODULES = ['test1.spiders']
NEWSPIDER_MODULE = 'test1.spiders'


USER_AGENT = 'test1 (+http://www.yourdomain.com)'
ROBOTSTXT_OBEY = False


DOWNLOAD_DELAY = 1



REQUEST_FINGERPRINTER_IMPLEMENTATION = '2.7'
TWISTED_REACTOR = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'

DUPEFILTER_CLASS = "scrapy_redis.dupefilter.RFPDupeFilter"
SCHEDULER = "scrapy_redis.scheduler.Scheduler"
SCHEDULER_PERSIST = True

ITEM_PIPELINES = {
    'test1.pipelines.Test1Pipeline': 300,
    'scrapy_redis.pipelines.RedisPipeline': 400,
}

REDIS_HOST = '192.168.25.231'
REDIS_PROT = 6379