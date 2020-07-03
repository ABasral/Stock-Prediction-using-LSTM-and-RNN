from selenium import webdriver
from threading import Timer


def downloadDataset(link):

   url_mid = str(link)
   driver = webdriver.Chrome("/Users/lakshaymittal/PycharmProjects/ML_robot/Drivers/chromedriver")
   driver.minimize_window()



   url_first = 'https://www.bseindia.com/markets/equity/EQReports/StockPrcHistori.aspx?expandable=6&scripcode='
   url_last = '&flag=sp&Submit=G'
   url = url_first + url_mid + url_last

   driver.get(url)
   driver.maximize_window()

   driver.execute_script(
      "document.getElementById('ContentPlaceHolder1_txtFromDate').setAttribute('value', '01/01/2010')"
   )

   driver.find_element_by_id('ContentPlaceHolder1_btnDownload').click()
   driver.minimize_window()



   close = Timer(8.0, driver.close())
