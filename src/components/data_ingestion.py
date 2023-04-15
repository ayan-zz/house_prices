import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import datatransfomation,datatransformationconfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig
#initialize the data ingestion configuration
@dataclass
class dataingestionconfig:
    train_path:str=os.path.join('artifacts','train.csv')
    test_path:str=os.path.join('artifacts', 'test.csv')
    raw_path:str=os.path.join('artifacts', 'raw.csv')
  

class dataingestion:
    def __init__(self):
        self.ingestion_config=dataingestionconfig()
    
    def initiate_dataingestion(self):
        logging.info('start data ingestion')
        try:
            df=pd.read_csv('https://raw.githubusercontent.com/krishnaik06/FSDSRegression/main/notebooks/data/gemstone.csv')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_path, index=False)
            logging.info("train test split")
            train_set,test_set=train_test_split(df, test_size=0.30, random_state=40)

            train_set.to_csv(self.ingestion_config.train_path, index=False)
            test_set.to_csv(self.ingestion_config.test_path,index =False)

            logging.info('data ingestion is complete')

            return(self.ingestion_config.train_path,self.ingestion_config.test_path)


        except Exception as e:
            logging.info("error occured at data ingestion stage")
            raise CustomException (e,sys)

if __name__=="__main__":
    obj=dataingestion()
    train_data,test_data=obj.initiate_dataingestion()

    data_trans=datatransfomation()
    train_arr,test_arr,_=data_trans.initiate_data_transformation(train_data,test_data)

    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)
    


