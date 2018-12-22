from bs4 import BeautifulSoup
import urllib.request
import csv
import re
from pprint import pprint

ThoiSu = "https://vnexpress.net/thoi-su"
TheThao = "https://vnexpress.net/the-thao"
TheGioi = "https://vnexpress.net/the-gioi"

def loadPage(url):
    try:
        page = urllib.request.urlopen(url)
        if(page.code == 200):
            soup = BeautifulSoup(page, 'html.parser')
            return soup
    except:
        return ""
    
    return ""

def getNewFeeds(soup):
    sideBar = soup.find('section', class_='sidebar_1')

    if(sideBar):
        return sideBar.find_all('article', class_='list_news')

    return []

def getContentUrl(feed):
    title = feed.find("h3", class_="title_news")
    if(title):
        a = title.find("a")
        if(a):
            return a.get('href')
    return ""

def getNextPage(soup):
    pagination = soup.find(
        'div', class_='pagination')
    
    if(pagination):
        nextP = pagination.find(
            'a', class_='next')
        if(nextP):
            return "https://vnexpress.net" + nextP.get('href')
    
    return ""

def crawlContent(url):
    soup = loadPage(url)
    if(soup):
        contentTag = soup.find("section", class_="sidebar_1")
        content = ""
        if(contentTag): 
            content = ""
            headerTag = contentTag.find("h1", class_="title_news_detail")
            descriptionTag = contentTag.find("h2", class_="description")
            contentTag = contentTag.find("article", class_="content_detail")
            if(headerTag):
                content = content + " " + headerTag.getText()
            if(descriptionTag):
                content = content + " " + descriptionTag.getText()
            if(contentTag):
                content = content + " " + contentTag.getText()
            
        return re.sub(' +', ' ', re.sub('\r?\n', " ", content))
    return ""

def crawlNewFeeds(soup, maxFeed = 5, numFeed = 0, array = []):

    if(soup):
        if(numFeed >= maxFeed):
            return array
        
        nextPage = getNextPage(soup)
        newFeeds = getNewFeeds(soup)

        for feed in newFeeds:
            if(numFeed >= maxFeed):
                return array
            numFeed = numFeed + 1
            url = getContentUrl(feed)
            if(url):
                content = crawlContent(url)
                if(content):
                    array.append(content)
        
        if(nextPage):
            return crawlNewFeeds(loadPage(nextPage), maxFeed, numFeed, array)  

    return array 

def crawlNews(url, label, fileName = "data.csv", numFeed = 100):
    print("writing " + label)
    with open(fileName, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['label', 'text'])
        array = crawlNewFeeds(loadPage(url), numFeed)

        for text in array:
            writer.writerow([label, text])
    print("done " + label)
    print("======================================")
    return

def crawler(data = []):

    with open("datasets.csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['label', 'text'])

        for info in data:
            label = info[0]
            url = info[1]
            numFeed = info[2]
            print("writing " + label + " " + url)
            with open(label + ".csv", 'w') as _csv_file:
                wr = csv.writer(_csv_file)
                wr.writerow(['label', 'text'])

                array = crawlNewFeeds(loadPage(url), numFeed)
                for text in array:
                    writer.writerow([label, text])
                    wr.writerow([label, text])
            print("done " + label)
            print("======================================")
    return

# crawlNews(ThoiSu, "ThoiSu", "ThoiSu.csv", 200)
# crawlNews(TheThao, "TheThao", "TheThao.csv", 200)
# crawlNews(TheGioi, "TheGioi", "TheGioi.csv", 200)

crawler([
    ["ThoiSu", ThoiSu, 200],
    ["TheThao", TheThao, 200],
    ["TheGioi", TheGioi, 200],
])