from selenium import webdriver
from threading import Timer


def downloadDataset(link):

   url_mid = str(link)
   driver = webdriver.Chrome("/Stock-Prediction-using-CNN/Drivers/chromedriver")
   driver.minimize_window()



   url_first = 'https://www.bseindia.com/markets/equity/EQReports/StockPrcHistori.aspx?expandable=6&scripcode='
   url_last = '&flag=sp&Submit=G'
   url = url_first + url_mid + url_last

   driver.get(url)
   driver.maximize_window()
   # driver.find_element_by_id("ContentPlaceHolder1_txtFromDate").send_keys('')
   # driver.find_element_by_id("ContentPlaceHolder1_txtFromDate").send_keys('01012010')
   # driver.find_element_by_id("ContentPlaceHolder1_txtToDate").send_keys('')
   # driver.find_element_by_id("ContentPlaceHolder1_txtToDate").send_keys('13/04/2020')
   # driver.find_element_by_id('ContentPlaceHolder1_btnSubmit').click()
   # ABOVE CODE IS NOT NEEDED

   driver.execute_script(
      "document.getElementById('ContentPlaceHolder1_txtFromDate').setAttribute('value', '01/01/2010')"
   )

   driver.find_element_by_id('ContentPlaceHolder1_btnDownload').click()
   driver.minimize_window()


   #t = Timer(5.0, shutil.move('/Users/lakshaymittal/Downloads/' + url_mid + '.csv',
      #                        '/Users/lakshaymittal/PycharmProjects/ML_robot/Datasets')
    #         )

   close = Timer(8.0, driver.close())
