import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait, Select
from bs4 import BeautifulSoup as Soup
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome('/Users/vigneshnatarajan/Downloads/chromedriver')

driver.get('https://nces.ed.gov/collegenavigator/')

WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="LeftContent"]/div[2]/div[6]')))

schools, grad_4_10, grad_4_12, grad_6_10, grad_6_12 = [], [], [], [], []
for count in range(1, 52):
    states = Select(driver.find_element_by_xpath('//*[@id="ctl00_cphCollegeNavBody_ucSearchMain_ucMapMain_lstState"]'))
    states.deselect_by_index(0)
    states.select_by_index(count)

    bachelors = driver.find_element_by_xpath('//*[@id="ctl00_cphCollegeNavBody_ucSearchMain_chkBach"]')
    bachelors.click()
    time.sleep(2)

    four_year = driver.find_element_by_xpath('//*[@id="ctl00_cphCollegeNavBody_ucSearchMain_chkLevelFourYear"]')
    four_year.click()
    time.sleep(2)

    results = driver.find_element_by_xpath('//*[@id="ctl00_cphCollegeNavBody_ucSearchMain_btnSearch"]')
    results.click()
    time.sleep(2)

    page_source = driver.page_source
    soup = Soup(page_source)
    results_text = 'Showing All Results'
    if (results_text in driver.page_source):
        page_count = [1]
    else:
        page = soup.find(id='ctl00_cphCollegeNavBody_ucResultsMain_divPagingControls')
        page_count = page.find_all('a')

    for page in range(1, len(page_count)+1):
        if page > 1:
            xpath_page = '//*[@id="ctl00_cphCollegeNavBody_ucResultsMain_divPagingControls"]/a[{}]'.format(page - 1)
            driver.find_element_by_xpath(xpath_page).click()
            time.sleep(3)
            soup1 = Soup(driver.page_source)
            schools1 = soup1.find_all('tr',class_='resultsW')
            schools2 = soup1.find_all('tr',class_='resultsY')
            total_count = len(schools1)+len(schools2)
            for i in range(1,total_count+1):
                xpath_school = '//*[@id="ctl00_cphCollegeNavBody_ucResultsMain_tblResults"]/tbody/tr[{}]/td[2]/a'.format(i)
                driver.find_element_by_xpath(xpath_school).click()
                soup2 = Soup(driver.page_source)
                school_name = soup2.find(class_='headerlg')
                test_text = "Bachelor’s degree graduation rates measure the percentage of entering students beginning their studies full-time and are planning to get a bachelor’s degree and who complete their degree program within a specified amount of time."
                test_text1 = "Bachelor’s degree graduation rates measure the percentage of entering students beginning their studies full-time and seeking a bachelor's degree, who earn a bachelor’s degree within a specified amount of time. "
                time.sleep(5)
                if ((test_text not in driver.page_source) and (test_text1 not in driver.page_source)):
                    driver.back()
                else:
                    div_tag = soup2.find(id='divctl00_cphCollegeNavBody_ucInstitutionMain_ctl05')
                    table = div_tag.find_all('table', class_='graphtabs')
                    headings = []
                    for head in table:
                        headings.append(head.find('tr').get_text())

                    index = headings.index("Graduation Rates for Students Pursuing Bachelor's Degrees")
                    img = table[index].find('img')
                    text = img['alt'].split()
                    if '5-year:' in text:
                        index1 = text.index('4-year:')
                        schools.append(school_name.get_text())
                        grad_4_10.append('0%')
                        grad_4_12.append(text[(index1+1):][2])
                        grad_6_10.append('0%')
                        grad_6_12.append(text[(index1+1):][4])
                    else:
                        beginning = [i for i,j in enumerate(text) if j == '[Began']
                        index1, index2, index3 = beginning[0], text.index('8-year:'), beginning[1]
                        if 'Fall' not in text:
                            schools.append(school_name.get_text())
                            grad_4_10.append(text[(index1+3):index2][1])
                            grad_4_12.append(text[(index3+3):][1])
                            grad_6_10.append(text[(index1+3):index2][3])
                            grad_6_12.append(text[(index3+3):][3])
                        else:
                            schools.append(school_name.get_text())
                            grad_4_10.append(text[(index1+4):index2][1])
                            grad_4_12.append(text[(index3+4):][1])
                            grad_6_10.append(text[(index1+4):index2][3])
                            grad_6_12.append(text[(index3+4):][3])
                    driver.back()
                time.sleep(3)
            driver.back()
        else:
            schools1 = soup.find_all('tr',class_='resultsW')
            schools2 = soup.find_all('tr',class_='resultsY')
            total_count = len(schools1)+len(schools2)
            for i in range(1, total_count+1):
                xpath_school = '//*[@id="ctl00_cphCollegeNavBody_ucResultsMain_tblResults"]/tbody/tr[{}]/td[2]/a'.format(i)
                driver.find_element_by_xpath(xpath_school).click()
                soup2 = Soup(driver.page_source)
                school_name = soup2.find(class_='headerlg')
                test_text = "Bachelor’s degree graduation rates measure the percentage of entering students beginning their studies full-time and are planning to get a bachelor’s degree and who complete their degree program within a specified amount of time."
                test_text1 = "Bachelor’s degree graduation rates measure the percentage of entering students beginning their studies full-time and seeking a bachelor's degree, who earn a bachelor’s degree within a specified amount of time. "
                time.sleep(5)
                if ((test_text not in driver.page_source) and (test_text1 not in driver.page_source)):
                    driver.back()
                else:
                    div_tag = soup2.find(id='divctl00_cphCollegeNavBody_ucInstitutionMain_ctl05')
                    table = div_tag.find_all('table', class_='graphtabs')
                    headings = []
                    for head in table:
                        headings.append(head.find('tr').get_text())

                    index = headings.index("Graduation Rates for Students Pursuing Bachelor's Degrees")
                    img = table[index].find('img')
                    text = img['alt'].split()
                    if '5-year:' in text:
                        index1 = text.index('4-year:')
                        schools.append(school_name.get_text())
                        grad_4_10.append('0%')
                        grad_4_12.append(text[(index1+1):][2])
                        grad_6_10.append('0%')
                        grad_6_12.append(text[(index1+1):][4])
                    else:
                        beginning = [i for i,j in enumerate(text) if j == '[Began']
                        index1, index2, index3 = beginning[0], text.index('8-year:'), beginning[1]
                        if 'Fall' not in text:
                            schools.append(school_name.get_text())
                            grad_4_10.append(text[(index1+3):index2][1])
                            grad_4_12.append(text[(index3+3):][1])
                            grad_6_10.append(text[(index1+3):index2][3])
                            grad_6_12.append(text[(index3+3):][3])
                        else:
                            schools.append(school_name.get_text())
                            grad_4_10.append(text[(index1+4):index2][1])
                            grad_4_12.append(text[(index3+4):][1])
                            grad_6_10.append(text[(index1+4):index2][3])
                            grad_6_12.append(text[(index3+4):][3])
                    driver.back()
                time.sleep(3)

    driver.back()
    driver.refresh()
    time.sleep(30)
