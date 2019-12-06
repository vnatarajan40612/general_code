import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup as Soup
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.action_chains import ActionChains

driver = webdriver.Chrome('/Users/vigneshnatarajan/Downloads/chromedriver')

driver.get('https://launchmycareer.utahfutures.org/')
soup_main = Soup(driver.page_source)

school_list_tags = soup_main.find('div', class_='col-xs-12 col-md-4 card-container').find_all('div',class_='card')[3].find('ol', class_='homepage-card-list').find_all('li')

school, degree, major, average_salary = [],[],[],[]
for count in range(1,len(school_list_tags)+1):
    xpath_school = '//*[@id="main"]/div[2]/div[2]/div[4]/div[2]/ol/li[{}]/a'.format(count)
    school_click = driver.find_element_by_xpath(xpath_school)
    school_click.click()
    time.sleep(4)
    
    soup = Soup(driver.page_source)
    major_list = soup.find('ul', class_='col-xs-12 dropdown-menu').find_all('li')

    for major_count in range(1,len(major_list)+1):
        select_major = driver.find_element_by_xpath('//*[@id="roi-carousel"]/div/div[1]/div[2]/form/div/a/span[1]')
        select_major.click()
        time.sleep(5)
        
        major_element = '/html/body/ul/li[{}]'.format(major_count)
        major_click = driver.find_element_by_xpath(major_element)
        actions = ActionChains(driver)
        actions.move_to_element(major_click).click(major_click).perform() 

        soup_submain = Soup(driver.page_source)
        text = "We're sorry, but something went wrong."

        if text in driver.page_source:
            driver.back()
            driver.refresh()

        else:
            degree_type = soup_submain.find('div', class_='degree-title title h6 text-grey-md').text.replace('\n',"").strip()
            degree.append(degree_type)

            major_type = soup_submain.find('div', class_='major-title title h4 text-grey-md').find('a').text.replace('\n',"").strip()
            major.append(major_type)

            salary = soup_submain.find('h4', class_='text-blue price').get_text()
            average_salary.append(salary)

            school_name = school_list_tags[count-1].find('a').get_text()
            school.append(school_name)
            time.sleep(5)

            driver.back()
            driver.refresh()

    driver.back()
    driver.refresh()
    time.sleep(4)
