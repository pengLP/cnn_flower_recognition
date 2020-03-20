# _*_ coding:utf-8 _*_
''''''
'''
    1.通过关键字进入图片界面
    
    2.加载图片
    queryWord:可爱图片
    word:可爱图片    
    pn:60
    gsm:3c
'''
import requests
import json
import time
import os

#要修改的参数列表
queryWord=input('请输入您要搜索的图片：')
pn=0
gsm=str(hex(pn))[-2:]
timestrp=int(time.time()*1000)
#num表示照片数量
num=1
#while实现类似翻页功能，遍历所有图片信息
while True:
    #请求的url
    url='https://image.baidu.com/search/acjson?' \
        'tn=resultjson_com&ipn=rj&ct=201326592&' \
        'is=&fp=result&queryWord={0}&cl=2&lm=-1&ie=utf-8&' \
        'oe=utf-8&adpicid=&st=-1&z=&ic=0&word={0}&s=&se=' \
        '&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&pn={1}&rn=30&gsm={2}&{3}='
    #伪装头部
    header={
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.119 Safari/537.36'
    }
    #解析为json()语句
    try:
        r_mus=requests.get(url.format(queryWord,pn,gsm,timestrp),headers=header).json()
    except BaseException as e:
        print("此处有错误%s"%e)
    print(r_mus)
    #遍历每一张图片信息
    for image in r_mus['data']:
        if image:
            #获取图片地址
            i_url=image['middleURL']
            #请求该地址
            r_img=requests.get(i_url,headers=header,stream=True).raw.read()
            print('正在读取第{}张图片'.format(num))
            num+=1
            time.sleep(0.7)
            #创建pictures目录
            if os.path.exists('data/other/'):
                pass
            else:
                os.mkdir('data/other/')
            #保存图片到文件夹pictures
            with open('data/other/'+str(int(time.time()))+'.jpg','wb')as files:
                files.write(r_img)
    listNum = r_mus['listNum']
    if listNum>pn:
        pn+=30
        gsm = str(hex(pn))[-2:]
        time.sleep(5)
    else:
        break