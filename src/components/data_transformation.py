import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):

        """This function is responsible for data transformation"""

        logging.info("Data Transformation method has started")
        try:
            num_features = ['reading score','writing score']
            cat_features = ['gender','race/ethnicity','parental level of education',
                            'lunch','test preparation course'] 
            

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical and Categorical Pipeline completed")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_features),
                    ('cat_pipeline',cat_pipeline,cat_features) 
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining Preprocessor object")
            preprocessor_obj = self.get_data_transformer()

            target_column_name = 'math score'
            numerical_columns = ['reading score','writing score']
            
            input_feature_train_df,target_feature_train_df = train_df.drop(columns=[target_column_name],axis=1),train_df[target_column_name]
            input_feature_test_df,target_feature_test_df = test_df.drop(columns=[target_column_name],axis=1),test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_df_transformed = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_df_transformed = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_df_transformed,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df_transformed,np.array(target_feature_test_df)]

            logging.info("Saved Preprocessor object")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)

