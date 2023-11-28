import yaml
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd

credential_pathway = "/Users/jaeheonlee/Desktop/CustomerLoans/exploratory-data-analysis---customer-loans-in-finance210/credentials.yaml"

def read_yaml():
    with open(credential_pathway,'r') as file:
        credentials = yaml.safe_load(file)
    return credentials

# Milestone 2
class RDSDatabaseConnector:
    # Milestone 2
    def __init__(self):
        self.credentials = read_yaml()
        self.engine = self.init_sqlalchemy_engine()
    
    # Milestone 2
    def init_sqlalchemy_engine(self):
        """
        Initialises the SQLAlchemy engine from the credentials provided from class.
        """
        sqlalchemy_engine = create_engine(f"postgresql://{self.credentials['RDS_USER']}:{self.credentials['RDS_PASSWORD']}@{self.credentials['RDS_HOST']}:{self.credentials['RDS_PORT']}/{self.credentials['RDS_DATABASE']}")
        return sqlalchemy_engine
    
    # Milestone 2
    def extract_RDS(self):
        """
        Extracts the data from the RDS database 
        Output: returns a Pandas data frame of the table.
        """
        table_name = "loan_payments"
        df = pd.read_sql(f"SELECT * FROM {table_name};", self.engine)
        return df
    
    # Milestone 2
    def save_to_csv(self, df):
        """
        Saves the data to local machine as .csv format
        Output: .csv file of the dataframe
        """
        return df.to_csv('loan_payments.csv', index = False)
    


if __name__ == "__main__":
    """
    The main function creates an instance of the class, then extracts the data from the RDS database,
    then saves the database into a csv file locally.
    Once the csv file is saved locally, the last two lines can be commented out as a local copy has been created already.
    """
    db_connector = RDSDatabaseConnector()
    df_loan_payments = db_connector.extract_RDS()
    db_connector.save_to_csv(df_loan_payments)