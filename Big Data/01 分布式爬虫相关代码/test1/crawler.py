import requests
import time
import xlwt
import tqdm

#parameters
sleeptime = 3
regionMax = 1

#结果保存path
savepath = "./newdata.xls"

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

#构造爬取点评信息头部
headers = {
    'Host':'www.ly.com',
    'Connection':'close',
    'Cache-Control':'max-age=0',
    'Upgrade-Insecure-Requests':'1',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36',
    'Sec-Fetch-Mode':'navigate',
    'Sec-Fetch-User':'?1',
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
    'Sec-Fetch-Site':'none',
    'Accept-Encoding':'gzip, deflate, br',
    'Accept-Language':'zh-CN,zh;q=0.9',
    'Cookie':'__tctma=144323752.1616121741862714.1616121741804.1616121741804.1616121741804.1;_dx_uzZo5y=44f8cfaca8713500850104826156abf49892e4440aae52868c9cf006ffbef7be7a9ba962;_dx_app_bc4b3ca6ae27747981b43e9f4a6aa769=60540f8f0xJbitehqkQspGmZCoRJCl9YshYbnYQ1; ASP.NET_SessionId=h4cui2szf5i2ika5rsyfsdgz; NewProvinceId=16; NCid=226; NewProvinceName=%E6%B1%9F%E8%8B%8F; NCName=%E8%8B%8F%E5%B7%9E; qdid=-9999; 17uCNRefId=RefId=0&SEFrom=&SEKeyWords=; TicketSEInfo=RefId=0&SEFrom=&SEKeyWords=; CNSEInfo=RefId=0&tcbdkeyid=&SEFrom=&SEKeyWords=&RefUrl=; Hm_lvt_c6a93e2a75a5b1ef9fb5d4553a2226e5=1616121746,1616122731; Hm_lpvt_c6a93e2a75a5b1ef9fb5d4553a2226e5=1616122731; __tctmc=144323752.106185093; __tctmd=144323752.737325; __tctmb=144323752.2693123087507125.1616121741804.1616122726589.2; __tctmu=144323752.0.0; __tctmz=144323752.1616122726589.1.1.utmccn=(direct)|utmcsr=(direct)|utmcmd=(none); longKey=1616121741862714; __tctrack=0; route=8b01b73ddb9a0b35bfc0aec7417be66a',
}

#session参数
s = requests.session()
s.headers = headers
requests.DEFAULT_RETRIES = 5
s.keep_alive = False

#爬取点评信息
results = []
resdp = []
for sid in tqdm.tqdm(sidlist):
    for i in tqdm.tqdm(range(0,999)):
        try:
            time.sleep(sleeptime)
            url = 'https://www.ly.com/scenery/AjaxHelper/DianPingAjax.aspx?action=GetDianPingList&sid='+sid+'&page='+str(i)+'&pageSize=10&labId=1&sort=0&iid=0.6401678021327853'
            comments = s.get(url=url,timeout=10,verify=True).text
            for j in range(1,11):
                results.append(comments.split('dpContent":"')[j].split('","dpDate')[0])
                dp = comments.split('"lineAccess":"')[j].split('","markColor"')[0]
                if dp == "好评":
                    dp = 1
                else:
                    dp = 0
                resdp.append(dp)
        except:
            break

#爬取结果保存
workbook = xlwt.Workbook(encoding="utf-8")
worksheet = workbook.add_sheet('sheet1')
worksheet.write(0,0,label='label')
worksheet.write(0,1,label='review')
for i in range(len(results)):
    worksheet.write(i+1,0,label=resdp[i])
    worksheet.write(i+1,1,label=results[i])
workbook.save(savepath)
