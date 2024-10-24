import time
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import requests
import csv



class CoinMarketCapCrawler:

    def __init__(self,chosenCoins,startdate,enddate):
        print(startdate,enddate)
        self.chosenCoins=chosenCoins
        self.startdate=startdate
        self.enddate=enddate

    def crawler(self):


        # Setting up the soup object

        html_text = requests.get('https://coinmarketcap.com/').text
        soup = BeautifulSoup(html_text, 'lxml')
        coinTable = soup.find_all('tr')

        # print(coinTable)

        coinLinks = []
        coinNames = []
        for i in range(1, 11):
            name = coinTable[i].find('div', class_="sc-16r8icm-0 sc-1teo54s-1 dNOTPP")
            coinNames.append(name.p.text)

        # print(coinNames)

        for i in range(11, len(coinTable)):
            detail = coinTable[i].find_all('td')
            # print(detail)
            detail = detail[2].find('a', class_="cmc-link")
            detail = detail.find_all('span')
            coinNames.append(detail[1].text)

        # Find out the links of the chosen Coins

        for i in range(len(self.chosenCoins)):
            for j in range(len(coinNames)):
                if self.chosenCoins[i] == coinNames[j]:
                    if j < 11:
                        detail = coinTable[j + 1].find('div', class_="sc-16r8icm-0 escjiH")
                        coinLinks.append(f"https://coinmarketcap.com{detail.a['href']}historical-data/")
                    else:
                        detail = coinTable[j + 1].find_all('td')
                        coinLinks.append(f"https://coinmarketcap.com{detail[2].a['href']}historical-data/")

        # print(coinLinks)
        opts = Options()
        opts.add_argument('--window-size=1920,1080')
        opts.add_argument('--headless')
        opts.add_argument('--no-sandbox')
        opts.add_argument('--disable-dev-shm-usage')
        opts.add_argument('--start-maximized')
        opts.add_argument('--ignore-certificate-errors')
        driver = webdriver.Chrome(executable_path="F:\Browsers Driver\chromedriver.exe")
        driver.maximize_window()

        for coin in range(len(coinLinks)):
            driver.get(coinLinks[coin])

            driver.implicitly_wait(15)
            driver.execute_script("window.scrollBy(0,500)", "")

            datebtn = driver.find_element(By.XPATH,
                                          "/html/body/div/div/div[1]/div[2]/div/div[3]/div/div/div[1]/span/button").click()
            # WebDriverWait(driver,30).until(EC.presence_of_element_located((By.XPATH,'/html/body/div/div/div[1]/div[2]/div/div[3]/div/div/div[1]/div/div/div[1]/div/div/div[1]/div[1]/div/div/div[2]/div[1]/div[1]/span[2]')))

            driver.find_element(By.XPATH,
                                '/html/body/div/div/div[1]/div[2]/div/div[3]/div/div/div[1]/div/div/div[1]/div/div/div[1]/div[1]/div/div/div[2]/div[1]/div[1]/span[2]').click()
            driver.find_element(By.XPATH,
                                '/html/body/div/div/div[1]/div[2]/div/div[3]/div/div/div[1]/div/div/div[1]/div/div/div[1]/div[1]/div/div/div[2]/div[1]/div[1]/span[2]').click()

            while True:
                html_text_new_page = driver.page_source

                new_soup = BeautifulSoup(html_text_new_page, 'lxml')
                years = new_soup.find('div', class_="yearpicker show")
                years = years.find_all('span')
                if int(self.startdate) >= int(years[0].text):

                    yearElementsXpath = []
                    for i in range(1, 13):
                        yearElementsXpath.append(
                            f"/html/body/div/div/div[1]/div[2]/div/div[3]/div/div/div[1]/div/div/div[1]/div/div/div[1]/div[1]/div/div/div[2]/div[1]/div[1]/div[2]/span[{i}] ")

                    for year in range(len(years)):
                        if self.startdate == years[year].text:
                            driver.find_element(By.XPATH, yearElementsXpath[year]).click()
                            driver.find_element(By.XPATH,
                                                "/html/body/div/div/div[1]/div[2]/div/div[3]/div/div/div[1]/div/div/div[1]/div/div/div[1]/div[1]/div/div/div[2]/div[1]/div[1]/div[1]/span[1]").click()

                            # Find out the first day of the month
                            html_text_new_page = driver.page_source
                            new_soup = BeautifulSoup(html_text_new_page, 'lxml')
                            days = new_soup.find('div', class_="react-datepicker__week")
                            days = days.find_all('div')

                            for day in range(len(days)):

                                if days[day].text == "1":
                                    driver.find_element(By.XPATH,
                                                        f"/html/body/div/div/div[1]/div[2]/div/div[3]/div/div/div[1]/div/div/div[1]/div/div/div[1]/div[1]/div/div/div[2]/div[2]/div[1]/div[{day + 1}]").click()
                                    break

                            # Checking for the End date
                            for click in range(((int(self.enddate) - int(self.startdate)) + 1) * 12):
                                driver.find_element(By.XPATH,
                                                    "/html/body/div/div/div[1]/div[2]/div/div[3]/div/div/div[1]/div/div/div[1]/div/div/div[1]/div[1]/div/div/div[2]/div[1]/div[1]/span[3]").click()

                            # Choose the last day
                            html_text_new_page = driver.page_source
                            # print(html_text_new_page)

                            new_soup = BeautifulSoup(html_text_new_page, 'lxml')

                            days = new_soup.find('div', class_="react-datepicker__week")
                            days = days.find_all('div')
                            for day in range(len(days)):
                                if days[day].text == "1":
                                    driver.find_element(By.XPATH,
                                                        f"/html/body/div/div/div[1]/div[2]/div/div[3]/div/div/div[1]/div/div/div[1]/div/div/div[1]/div[1]/div/div/div[2]/div[2]/div[1]/div[{day + 1}]").click()
                                    break

                            driver.find_element(By.XPATH,
                                                "/html/body/div/div/div[1]/div[2]/div/div[3]/div/div/div[1]/div/div/div[1]/div/div/div[2]/span/button").click()
                    break



                else:
                    driver.find_element(By.XPATH,
                                        "/html/body/div/div/div[1]/div[2]/div/div[3]/div/div/div[1]/div/div/div[1]/div/div/div[1]/div[1]/div/div/div[2]/div[1]/div[1]/span[1]").click()

            # Gathering The Data and save it in a csv file
            time.sleep(2)
            driver.implicitly_wait(15)
            html_config_page = driver.page_source
            config_soup = BeautifulSoup(html_config_page, 'lxml')
            # print(html_config_page)
            tableData = config_soup.find_all('tr')
            # print(tableData)

            # Open the csv file to save the data
            with open(
                    f"Dataset/{self.chosenCoins[coin]}_Coinmarketcap_{self.startdate}_{self.enddate}.csv",
                    'w', encoding="utf-8",
                    newline='') as f:
                fieldNames = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap', 'Currency']
                wr = csv.DictWriter(f, fieldnames=fieldNames)

                wr.writeheader()

                # loop over all rows
                for row in range(1, len(tableData)):
                    data = tableData[row].find_all('td')
                    # print(data)

                    wr.writerow(
                        {'Date': f"{data[0].text}", 'Open': f"{str(data[1].text).strip('$').replace(',', '')}",
                         'High': f"{str(data[2].text).strip('$').replace(',', '')}",
                         'Low': f"{str(data[3].text).strip('$').replace(',', '')}",
                         'Close': f"{str(data[4].text).strip('$').replace(',', '')}",
                         'Volume': f"{str(data[5].text).strip('$').replace(',', '')}",
                         'Market Cap': f"{str(data[6].text).strip('$').replace(',', '')}", 'Currency': "$"})

        driver.close()


