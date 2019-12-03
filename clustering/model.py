import os
import sys
import numpy as np
import pandas as pd
import re
from scipy.stats import lognorm
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from sklearn import metrics

class parameter_estimation(object):
    def __init__(self, exclude_FDR = False, salary_growth_outlier_weight = 0.1):
        
        # Read data
        self.data = pd.read_csv('data_cleaned/main_data.txt', sep = '\t')  
        self.school_clustering = pd.read_csv('data_cleaned/school_clustering.txt', sep='\t')
        self.major_list = pd.read_csv('data_cleaned/major_list.txt', sep='\t')
        self.salary_growth = pd.read_csv('data_cleaned/salary_growth_data.txt', sep = '\t')
        self.salary_growth_outlier_weight = salary_growth_outlier_weight

        # Clean columns names
        self.data.columns = [x.lower() for x in self.data.columns]
        self.salary_growth.columns = [x.lower() for x in self.salary_growth.columns]

        if exclude_FDR:
            self.data = self.data[self.data['source']!='FDR Report'].copy()

        # Select data that are in the clustering data
        self.data = self.data[self.data['school_in_clustering'] == 'Y']
        self.salary_growth = self.salary_growth[self.salary_growth['school_in_clustering'] == 'Y']
        # Get sigma calculated from 25 - 75 percentile or Average - Median
        self.data['sigma_qt'] = self.data.apply(lambda row: self._sigma_qt(row),axis=1)
        # This sigma value is used in calculation from median from mean
         
        medain_average_ratio_t = self.data.query(''' median_salary>0 and average_salary>0 ''')[['median_salary','average_salary']].median()
        self.medain_average_ratio = medain_average_ratio_t[0]/medain_average_ratio_t[1]

        # Get salary median estimated
        self.data['salary_median'] = self.data.apply (lambda row: self._median(row, self.medain_average_ratio),axis=1)
        
        # Identify the schools that have overall records only
        only_all_schools = set(self.data.loc[(pd.isna(self.data['majorcategoryid'])),'school_name_matched']) - set(self.data.loc[(~pd.isna(self.data['majorcategoryid'])),'school_name_matched'])
        self.data.loc[self.data['school_name_matched'].isin(only_all_schools), 'only_all_flag'] = 1
        self.data.fillna({'only_all_flag':0}, inplace=True)
        
        # Add salary growth match flag
        self.school_clustering.loc[self.school_clustering['school_name'].isin(self.salary_growth['school_name_matched']),'matched_flag_growth'] = 1
        self.school_clustering.fillna({'matched_flag_growth':0}, inplace=True)
        
        # Add salary growth else match flag
        self.school_clustering.loc[self.school_clustering['school_name'].isin(self.salary_growth.query('il_flag==0')['school_name_matched']),'matched_flag_growth_else'] = 1
        self.school_clustering.fillna({'matched_flag_growth_else':0}, inplace=True)
        
        # Add salary match flag
        self.school_clustering.loc[self.school_clustering['school_name'].isin(self.data.query('only_all_flag==0')['school_name_matched']),'matched_flag'] = 1
        self.school_clustering.fillna({'matched_flag':0}, inplace=True)
        
        # Add sigma match flag
        sigma_schools = list(self.data.loc[self.data['sigma_qt']>0,'school_name_matched'])
        self.school_clustering.loc[self.school_clustering['school_name'].isin(sigma_schools),'matched_flag_sigma'] = 1
        self.school_clustering.fillna({'matched_flag_sigma':0}, inplace=True)
        
        self.sigma_data = self.data.loc[self.data['sigma_qt']>0].copy()
        self.data.set_index('school_name_matched', inplace=True)
        self.salary_growth.set_index('school_name_matched', inplace=True)
        self.sigma_data.set_index('school_name_matched', inplace=True)
        
        ### create title_to_category_ratio
        self.data['major_category_median'] = self.data.groupby(['school_name_matched','state','school_name','major_category'])['salary_median'].transform(np.mean)
        self.data['ratio'] = self.data['major_category_median'] / self.data['salary_median']
        self.title_to_category_ratio = self.data[self.data['major_title'] != 'all'].groupby(['major_category','major_title'])['ratio'].mean().reset_index()
        self.title_to_category_ratio = pd.merge(self.major_list[['major_title','major_category']], self.title_to_category_ratio, on = ['major_title','major_category'], how = 'left')
        self.title_to_category_ratio = self.title_to_category_ratio.fillna(1)
        
        ### create category_to_school_ratio
        salary_median = self.data.loc[self.data['major_category'] == 'all'].reset_index().groupby('school_name_matched')['salary_median'].mean().reset_index(name = 'school_median')
        category_salary = self.data.loc[self.data['major_category'] != 'all',['major_category','major_title','salary_median']].reset_index().drop_duplicates(keep='first')
        school_median = pd.merge(salary_median, category_salary, on = 'school_name_matched', how = 'inner')
        school_median = school_median[school_median['major_title'] == 'all']
        school_median['ratio'] = school_median['salary_median'] / school_median['school_median']
        self.category_to_school_ratio = school_median.groupby('major_category')['ratio'].mean().reset_index()
        self.category_to_school_ratio = pd.merge(self.major_list[['major_category']].drop_duplicates(),self.category_to_school_ratio, on = ['major_category'], how = 'left')
        self.category_to_school_ratio = self.category_to_school_ratio.fillna(1)
        self.data = self.data.drop(columns = ['major_category_median','ratio'])
        
        ### Initialize functions
        self.build_tree()
        
    def _sigma_qt(self, row):
        if row['salary_min']!= 0:
            sigma = (np.log(row['salary_max']) - np.log(row['salary_min']))/(norm.ppf(0.75, loc=0, scale=1) - norm.ppf(0.25, loc=0, scale=1))
        elif row['average_salary'] != 0 and row['median_salary'] != 0 and row['average_salary'] > row['median_salary']:
            sigma = np.sqrt(2*(np.log(row['average_salary']) - np.log(row['median_salary'])))
        else:
            sigma = 0
        return sigma

    def _median (self, row, r):
        if row['median_salary'] != 0 :
            return row['median_salary']
        elif row['salary_min'] != 0:
            return np.exp((np.log(row['salary_min'])+np.log(row['salary_max']))/2)
        else:
            # return np.exp(np.log(row['average_salary']) - np.power(max_sigma,2)/2)
            return row['average_salary'] * r
    
    def build_tree(self):
        # features used for clustering
        clustering_features = ['state','level','control','long_x','lat_y','student_count','rank_num',\
                               'tuition','school_city_demo','school_city_gdp','matched_flag','matched_flag_growth_else','matched_flag_growth','matched_flag_sigma']

        clustering_data = self.school_clustering[clustering_features]

        # one hot encoding for numerical data
        cat_vars = ['state','level','control']

        for var in cat_vars:
            cat_list = pd.get_dummies(clustering_data[var], prefix=var)
            clustering_data1 =clustering_data.join(cat_list)
            clustering_data = clustering_data1

        clustering_data_vars = clustering_data.columns.values.tolist()
        to_keep = [i for i in clustering_data_vars if i not in cat_vars]
        clustering_data = clustering_data[to_keep]

        # scale the features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(clustering_data)
        clustering_data_trans = pd.DataFrame(scaler.transform(clustering_data), columns = clustering_data.columns)
        clustering_data_trans.index = self.school_clustering['school_name']

        self.data_train1 = (clustering_data_trans[clustering_data_trans['matched_flag'] == 1].drop(columns=['matched_flag','matched_flag_growth_else','matched_flag_growth','matched_flag_sigma']))
        self.data_train2 = (clustering_data_trans[clustering_data_trans['matched_flag_growth_else'] == 1].drop(columns=['matched_flag','matched_flag_growth_else','matched_flag_growth','matched_flag_sigma']))
        self.data_train3 = (clustering_data_trans[clustering_data_trans['matched_flag_growth'] == 1].drop(columns=['matched_flag','matched_flag_growth_else','matched_flag_growth','matched_flag_sigma']))
        self.data_train4 = (clustering_data_trans[clustering_data_trans['matched_flag_sigma'] == 1].drop(columns=['matched_flag','matched_flag_growth_else','matched_flag_growth','matched_flag_sigma']))
#         print(self.data_train.shape)
        self.data_test1 = (clustering_data_trans.drop(columns=['matched_flag','matched_flag_growth_else','matched_flag_growth','matched_flag_sigma']))
        self.data_test2 = (clustering_data_trans.drop(columns=['matched_flag','matched_flag_growth_else','matched_flag_growth','matched_flag_sigma']))
        self.data_test3 = (clustering_data_trans.drop(columns=['matched_flag','matched_flag_growth_else','matched_flag_growth','matched_flag_sigma']))
        self.data_test4 = (clustering_data_trans.drop(columns=['matched_flag','matched_flag_growth_else','matched_flag_growth','matched_flag_sigma']))
#         print(self.data_test.shape)
        
        self.kdt1 = KDTree(np.array(self.data_train1))
        self.kdt2 = KDTree(np.array(self.data_train2))
        self.kdt3 = KDTree(np.array(self.data_train3))
        self.kdt4 = KDTree(np.array(self.data_train4))
        
    def get_salary_neighbors(self, school_name, k = 3):
        a = np.expand_dims(np.array(self.data_test1.loc[school_name]), axis=0)
        _, ind_list = self.kdt1.query(a, k)  
        x = self.data_train1.iloc[ind_list[0,:]].index
        return list(x)
    
    def get_growth_neighbors_else(self, school_name, k = 3):
        a = np.expand_dims(np.array(self.data_test2.loc[school_name]), axis=0)
        _, ind_list = self.kdt2.query(a, k)  
        x = self.data_train2.iloc[ind_list[0,:]].index
        return list(x)
    
    def get_growth_neighbors(self, school_name, k = 3):
        a = np.expand_dims(np.array(self.data_test3.loc[school_name]), axis=0)
        _, ind_list = self.kdt3.query(a, k)  
        x = self.data_train3.iloc[ind_list[0,:]].index
        return list(x)

    def get_sigma_neighbors(self, school_name, k = 3):
        a = np.expand_dims(np.array(self.data_test4.loc[school_name]), axis=0)
        _, ind_list = self.kdt4.query(a, k)  
        x = self.data_train4.iloc[ind_list[0,:]].index
        return list(x)
    
    def find_one_median(self, school_name, major_title, major_category):
        matched_records = self.data.loc[[school_name]]

        ### if major_title is matched ###
        if major_title in list(matched_records.major_title):
#             print('find_major_title {}'.format(major_title))
            median = matched_records.loc[matched_records['major_title'] == major_title,'salary_median'].mean()
            salary_similar = list(matched_records.loc[matched_records['major_title'] == major_title,'salary_median'])
        ### if major_category is matched ###
        elif major_category in np.unique(matched_records.major_category):
            medain_temp = matched_records.loc[matched_records['major_category']==major_category,'salary_median'].mean()
#             print(medain_temp)
            ratio = self.title_to_category_ratio.loc[self.title_to_category_ratio['major_title'] == major_title,'ratio']
            median = medain_temp * float(ratio)
            salary_similar = list(matched_records.loc[matched_records['major_category']==major_category,'salary_median'])
        ### if major_category is not matched ###
        else:
            medain_temp = matched_records['salary_median'].mean()
#             print(medain_temp)
            ratio1 = self.category_to_school_ratio.loc[self.category_to_school_ratio['major_category']==major_category,'ratio']
            ratio2 = self.title_to_category_ratio.loc[self.title_to_category_ratio['major_title'] == major_title,'ratio']
            median = float(medain_temp) * float(ratio1) * float(ratio2)
            salary_similar = list(matched_records['salary_median'])

        return float(median), salary_similar

    
    def find_one_sigma(self, school_name, major_title, major_category):
        matched_records = self.sigma_data.loc[[school_name]]
        
        if major_title in list(matched_records.major_title):
            sigma = matched_records.loc[matched_records['major_title'] == major_title, 'sigma_qt'].mean()
        
        elif major_category in np.unique(matched_records.major_category):
            sigma = matched_records.loc[matched_records['major_category'] == major_category, 'sigma_qt'].mean()
        
        else:
            sigma = matched_records['sigma_qt'].mean()
            
        return sigma

    def get_value(self, school, m_title, k):
        # first get the major_category and school state from input
        m_category, state = self.match_input(school, m_title)
#         print(school)
#         print(m_title)
#         print(m_category)
        # if we have school's salary information:
        similar_schools = self.get_salary_neighbors(school, k)
        self.salary_similar_schools = similar_schools
        similar_schools_sigma = self.get_sigma_neighbors(school, k)
        # print(similar_schools)
        median_array = []
        similar_median_array = []
        sigma_array = []
        for s1 in similar_schools:
            a,b = self.find_one_median(s1, m_title, m_category)
            median_array.append(a)
            similar_median_array.append(b)

        for s2 in similar_schools_sigma:
            sigma_array.append(self.find_one_sigma(s2, m_title, m_category))
#         print(median_array)
        
        median_knn = np.mean(np.apply_over_axes(np.sort, np.array(median_array), axes=0)[:2])
        sigma_knn = np.mean(sigma_array)
        
        if school in self.data[self.data['only_all_flag']==0].index:
            median_self, _ = self.find_one_median(school, m_title, m_category)
            median = 0.8 * median_self + 0.2 * median_knn
        else:
            median = median_knn

        if school in self.sigma_data.index:
            sigma_self = self.find_one_sigma(school, m_title, m_category)
            sigma = 0.8 * sigma_self + 0.2 * sigma_knn
        else:
            sigma = sigma_knn
        
        return median, sigma, similar_median_array
    
    def find_growth(self, school_name, major_title, major_category):
        matched_records = self.salary_growth.loc[[school_name]]
        
        if major_title in list(matched_records.major_title):
            growth = list(matched_records.loc[matched_records['major_title'] == major_title, ['growth_rate_2','growth_rate_3','growth_rate_4','growth_rate_5']].mean())
        
        elif major_category in np.unique(matched_records.major_category):
            growth = list(matched_records.loc[matched_records['major_category'] == major_category, ['growth_rate_2','growth_rate_3','growth_rate_4','growth_rate_5']].mean())
        
        else:
            growth = list(matched_records[['growth_rate_2','growth_rate_3','growth_rate_4','growth_rate_5']].mean())
            
        return growth
            
    def get_growth(self, school, m_title):
        m_category, state = self.match_input(school, m_title)
        
        similar_schools = self.get_growth_neighbors(school, k = 3)
        similar_schools_else = self.get_growth_neighbors_else(school, k = 3)

        growth_array = []
        growth_array_else = []
        
        for ss in similar_schools:
            g1 = self.find_growth(ss, m_title, m_category)
            growth_array.append(g1)
        
        growth_array = np.array(growth_array)

        
        for sss in similar_schools_else:
            g2 = self.find_growth(sss, m_title, m_category)
            growth_array_else.append(g2)
        
        growth_array_else = np.array(growth_array_else)

        salary_growth_all = np.mean(growth_array, axis = 0)
        salary_growth_else = np.mean(growth_array_else, axis = 0)
        salary_growth = self.salary_growth_outlier_weight * salary_growth_all + (1-self.salary_growth_outlier_weight) * salary_growth_else
          
        return salary_growth
    
    def input_check(self,school):
        output = 1
        
        if school not in list(self.school_clustering['school_name']):
            print('Error: School name \"{}\" is not recorded!'.format(school))
            output = 0
        
        return output
    
    def match_input(self, school, major):
        mc = self.major_list.loc[self.major_list['major_title'] == major,'major_category'].values[0]
        state = np.array(self.school_clustering.loc[self.school_clustering['school_name']==school, 'state'])[0]
        return mc,state
    
    def find_estimate(self, school, majorID, k=5):
        
        school = school.lower()
        check_result = self.input_check(school)
        output = {}

        if check_result == 0:
                output['Error'] = {}
                output['Error']['salary_year_1'] = -1
                output['Error']['salary_year_2'] = -1
                output['Error']['salary_year_3'] = -1
                output['Error']['salary_year_4'] = -1
                output['Error']['salary_year_5'] = -1
                output['Error']['sigma'] = -1
                
        else: 

            if not isinstance(majorID, list):
                majorID = [majorID]
        
            for m in majorID:
                if int(m) < 1 or int(m) > max(self.major_list['majorID']):
                    print('Error: majorID not existed!')
                    output['Error'] = {}
                    output['Error']['salary_year_1'] = -1
                    output['Error']['salary_year_2'] = -1
                    output['Error']['salary_year_3'] = -1
                    output['Error']['salary_year_4'] = -1
                    output['Error']['salary_year_5'] = -1
                    output['Error']['sigma'] = -1
                    output['Error']['similar_schools'] = -1
                    output['Error']['similar_salary'] = -1
                    
                else:   
                    m_title = (self.major_list.loc[self.major_list['majorID'] == int(m), 'major_title'].values)[0]
                    output[m_title] = {}
                    median, sigma, similar_median = self.get_value(school, m_title, k)
                    salary_growth = self.get_growth(school, m_title)

                    output[m_title]['salary_year_1'] = median
                    output[m_title]['salary_year_2'] = output[m_title]['salary_year_1'] * (1 + salary_growth[0])
                    output[m_title]['salary_year_3'] = output[m_title]['salary_year_2'] * (1 + salary_growth[1])
                    output[m_title]['salary_year_4'] = output[m_title]['salary_year_3'] * (1 + salary_growth[2])
                    output[m_title]['salary_year_5'] = output[m_title]['salary_year_4'] * (1 + salary_growth[3])
                    output[m_title]['sigma'] = sigma
                    output[m_title]['similar_schools'] = self.salary_similar_schools
                    output[m_title]['similar_salary'] = similar_median
        
        return output