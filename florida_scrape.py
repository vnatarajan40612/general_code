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

driver.get('https://launchmycareerfl.org/')
soup_main = Soup(driver.page_source)
school_list_tags = soup_main.find('div', id='main').find('div', class_='container homepage').find('div', class_='col-xs-12 col-md-4 card-container').find_all('div', class_='card')[3].find('div', class_='row card-content').find('ol', class_='homepage-card-list').find_all('li')


school, degree, major, average_salary, percent_employed_yr1, percent_employed_yr5, percent_employed_ttl_yr1, percent_employed_ttl_yr5, continuing_education_yr1, continuing_education_yr5 = [],[],[],[],[],[],[],[],[],[]
for school_count in range(1,len(school_list_tags)+1):
    xpath_school_click = '//*[@id="main"]/div[2]/div[2]/div[4]/div[2]/ol/li[{}]/a'.format(school_count)
    school_click = driver.find_element_by_xpath(xpath_school_click)
    school_click.click()
    time.sleep(3)

    soup = Soup(driver.page_source)
    major_list = soup.find('ul', class_='col-xs-12 dropdown-menu')
    list_major = major_list.find_all('li')
    count_major = len(list_major)

    for major_count in range(1,(count_major+1)):
        select_major = driver.find_element_by_xpath('//*[@id="roi-carousel"]/div/div[1]/div[2]/form/div')
        select_major.click()
        time.sleep(6)

        xpath_major_click = '/html/body/ul/li[{}]'.format(major_count)
        major_click = driver.find_element_by_xpath(xpath_major_click)
        actions = ActionChains(driver)
        actions.move_to_element(major_click).click(major_click).perform()

        soup1 = Soup(driver.page_source)
        salary_average = soup1.find('h4', class_='text-blue price').get_text()
        average_salary.append(salary_average)
        school.append(school_list_tags[school_count-1].find('a').get_text())
        degree_type = soup1.find('div', class_='degree-title title h6 text-grey-md').text.replace('\n',"").strip()
        degree.append(degree_type)
        major_type = soup1.find('div', class_='major-title title h4 text-grey-md').find('a').text.replace('\n',"").strip()
        major.append(major_type)

        economic_success_div = soup1.find('div', class_='card text-center col-xs-12 col-sm-6 col-lg-4 hidden-xs visible-lg visible-md visible-sm')
        economic_success_2nd_div = economic_success_div.find('div', class_='card-wrapper')
        table = economic_success_2nd_div.find('table', class_='table')
        body_container = table.find('tbody')
        tr_tags = body_container.find_all('tr')
        td_tag_empl = tr_tags[0].find_all('td')
        percent_employed_yr1.append(td_tag_empl[0].get_text())
        percent_employed_yr5.append(td_tag_empl[1].get_text())
        td_tag_ttl = tr_tags[1].find_all('td')
        percent_employed_ttl_yr1.append(td_tag_ttl[0].get_text())
        percent_employed_ttl_yr5.append(td_tag_ttl[1].get_text())
        td_tag_ed = tr_tags[3].find_all('td')
        continuing_education_yr1.append(td_tag_ed[0].get_text())
        continuing_education_yr5.append(td_tag_ed[1].get_text())
        time.sleep(6)
        driver.back()
        driver.refresh()

    driver.back()
    time.sleep(6)
