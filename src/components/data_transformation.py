import os
import sys
import numpy  as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object
from src.logger import logging
from src.exception import CustomException

@dataclass
class datatransformationconfig:
    preprocessor_file_path=os.path.join('artifacts','preprocessor.pkl')

class datatransfomation:
    def __init__(self):
        self.data_transformation_obj=datatransformationconfig()
    
    
    def get_transformer_object(self):
        try:
            logging.info('initiation of data transformation')
            num_column=['carat', 'depth', 'table', 'x', 'y', 'z']
            cat_column=['cut', 'color', 'clarity']

            cut_category=['Ideal','Premium', 'Very Good', 'Good', 'Fair']
            color_category=['D','E','F', 'G', 'H', 'I','J']
            clarity_category=['VS2', 'SI2', 'VS1', 'SI1', 'IF', 'VVS2', 'VVS1', 'I1']

            num_pipeline=Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                             ('scaling',StandardScaler())
                       ])
            cat_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                             ('encoder',OrdinalEncoder(categories=[cut_category,color_category,clarity_category])),
                             ('scaling',StandardScaler())])
            
            preprocessor=ColumnTransformer([
                ("numerical_pipeline",num_pipeline,num_column),
                ("cat_pipeline",cat_pipeline,cat_column)])
            
            return preprocessor
        
        except Exception as e:
            logging.info('error in data transformation initiation')  
            raise  CustomException(e,sys)  
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            df_train=pd.read_csv(train_path)
            df_test=pd.read_csv(test_path)

            logging.info("Read Train ans test data completed")
            logging.info("obtaining preprocesser object")

            preprocessor_obj=self.get_transformer_object()

            target_column_name="price"
            drop_column=['price','id']
            
            target_feature_train_df=df_train[target_column_name]
            input_feature_train_df=df_train.drop(columns=drop_column,axis=1)
             

            target_feature_test_df=df_test[target_column_name]
            input_feature_test_df=df_test.drop(columns=drop_column,axis=1)
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)       

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_df] 

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_obj.preprocessor_file_path,
                obj=preprocessor_obj
            )            
            
            return (train_arr,test_arr,self.data_transformation_obj.preprocessor_file_path)

        except Exception as e:
            raise CustomException(e,sys)
        


        
