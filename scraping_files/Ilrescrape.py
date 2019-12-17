import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait, Select
from bs4 import BeautifulSoup as Soup
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome('/Users/vigneshnatarajan/Downloads/chromedriver')

driver.get('https://www.ilcollege2career.com/#/')

first_click = WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="tutorial-modal"]/div/div/div/div[3]/button[2]')))
first_click.click()

invisible = WebDriverWait(driver, 10).until(EC.invisibility_of_element_located((By.ID, 'tutorial-modal')))

if invisible:
    Year4_click = WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.ID, '4year')))
    Year4_click.click()

    count = 1
    university, major, average_salary, salary_growth = [], [], [], []
    while count < 29:
        select = Select(driver.find_element_by_xpath('//*[@id="program-1"]'))
        if count > 1:
            WebDriverWait(driver,10).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="program-1"]/option[3]')))
            select.select_by_visible_text('Agriculture & Related')

        select.select_by_index(count)
        results_click = WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="results-button"]')))
        results_click.click()

        WebDriverWait(driver,10).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="results-form"]/div[1]/div[2]/h2')))

        if count > 1:
            page_source = driver.page_source
            soup = Soup(page_source)
            major_value = soup.find(id='prgStudyID')
            uni = soup.find_all('a', attrs={'style':'text-decoration: underline; font-weight: bold'})
            salary_info = soup.find_all('td',class_='tableData table-first-program')
            growth_info = soup.find_all('td', attrs={'style':'position: relative; vertical-align: top; cursor:pointer; background-color:#f9f9f9'})

            for rows in uni:
                university.append(rows.get_text())

            for rows in salary_info:
                average_salary.append(rows.get_text())
                major.append(major_value.get_text())

            for counts, rows in enumerate(growth_info, start=1):
                if counts % 2 == 0:
                    salary_growth.append(rows.get_text())

        else:
            page_source = driver.page_source
            soup = Soup(page_source)
            major_value = soup.find(id='prgStudyID')
            uni = soup.find_all('a', attrs={'style':'text-decoration: underline; font-weight: bold'})
            salary_info = soup.find_all('td', attrs={'style':'position: relative; vertical-align: top; cursor:pointer; background-color:#fffdf0'})
            growth_info = soup.find_all('td', attrs={'style':'position: relative; vertical-align: top; cursor:pointer; background-color:#f9f9f9'})

            for rows in uni:
                university.append(rows.get_text())
                major.append(major_value.get_text())

            for counts, rows in enumerate(salary_info, start=1):
                if counts % 2 == 0:
                    average_salary.append(rows.get_text())

            for counts, rows in enumerate(growth_info, start=1):
                if counts % 2 == 0:
                    salary_growth.append(rows.get_text())

        count +=1
        time.sleep(8)
        driver.back()
        driver.refresh()
        WebDriverWait(driver,10).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="program-1"]')))
        time.sleep(3)
