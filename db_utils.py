import yaml
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime as dt
import os
import missingno as msno
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tabulate import tabulate
from statsmodels.graphics.gofplots import qqplot


credential_pathway = "/Users/jaeheonlee/Desktop/CustomerLoans/exploratory-data-analysis---customer-loans-in-finance210/credentials.yaml"
script_df = pd.read_csv('loan_payments.csv')
original_df = script_df.copy()

def read_yaml():
    with open(credential_pathway,'r') as file:
        credentials = yaml.safe_load(file)
    return credentials


class RDSDatabaseConnector:
    def __init__(self):
        self.credentials = read_yaml()
        self.engine = self.init_sqlalchemy_engine()

    def init_sqlalchemy_engine(self):
        """
        Initialises the SQLAlchemy engine from the credentials provided from class.
        """
        sqlalchemy_engine = create_engine(f"postgresql://{self.credentials['RDS_USER']}:{self.credentials['RDS_PASSWORD']}@{self.credentials['RDS_HOST']}:{self.credentials['RDS_PORT']}/{self.credentials['RDS_DATABASE']}")
        return sqlalchemy_engine
    
    def extract_RDS(self):
        """
        Extracts the data from the RDS database 
        """
        table_name = "loan_payments"
        df = pd.read_sql(f"SELECT * FROM {table_name};", self.engine)
        return df
    
    # Milestone 2
    def save_to_csv(self, df):
        """
        Saves the data to local machine as .csv format
        """
        return df.to_csv('loan_payments.csv', index = False)

if __name__ == "__main__":
    """
    The main function creates an instance of the class, then extracts the data from the RDS database,
    then saves the database into a csv file locally.
    Once the csv file is saved locally, the last two lines can be commented out as a local copy has been created already.
    """
    db_connector = RDSDatabaseConnector()
    #df_loan_payments = db_connector.extract_RDS()
    #db_connector.save_to_csv(df_loan_payments)


class DataTransform:
    def __init__(self, df):
        self.df = df
        
    def to_datetime(self, column_names):
        """
        Convert the column into a datetime type
        """
        for name in column_names:
            self.df[name] = pd.to_datetime(self.df[name], format = "%b-%Y")
    
    def format_month(self, column_names):
        """
        Extract the month from the format 36 months eg. 'term' column
        """
        for name in column_names:
            self.df[name] = self.df[name].str.extract('(\d+)').astype('float64')
            self.df.rename(columns = {name: name + " (in months)"}, inplace = True)
            
    def format_year(self, column_names):
        """
        Extract the month from the format 5 years eg. 'employment_length' column
        """
        for name in column_names:
            self.df[name] = self.df[name].str.extract('(\d+)').astype('float64')  
            self.df.rename(columns = {name: name + " (in years)"}, inplace = True)
            
    def to_category(self, column_names):
        """
        Convert the column into a category type
        """
        for name in column_names:
            self.df[name] = self.df[name].astype('category')


transformer = DataTransform(script_df)

to_datetime_columns = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']
to_datetime_month_columns = ['term']
to_datetime_year_columns = ['employment_length']
to_category_columns = ['home_ownership','verification_status', 'loan_status', 'payment_plan', 'purpose', 'application_type']

transformer.to_datetime(to_datetime_columns)
transformer.format_month(to_datetime_month_columns)
transformer.format_year(to_datetime_year_columns)
transformer.to_category(to_category_columns)

class DataFrameInfo:
    """
    This class extracts information fmo the dataframe and its columns
    """
    def __init__(self, df):
        self.df = df
    
    def describe_columns(self):
        """
        describe_columns: Describe all columns in the DataFrame to check their data types
        """
        return self.df.info()
    
    def get_statistical_values(self):
        """
        get_statistical_values: get median, standard deviation and mean from the columns and the DataFrame

        """
        return self.df.describe()
    
    def distinct_value_count(self, column_name):
        """
        distinct_value_count: count all distinct values in categorical columns

        """
        return self.df[column_name].value_counts(dropna = False)
    
    def print_dataframe_shape(self):
        """
        print_dataframe_shape: print out the shape of the dataframe
        """
        return self.df.shape
    
    def null_count(self):
        """
        null_count: count all null values in each column

        """
        return self.df.isna().sum()
    
    def null_percentage(self):
        """
        null_percentage: percentage of null values in each column
        """
        return (self.df.isna().sum() / len(self.df)) * 100        
    
    def dataframe_correlation_matrix(self):
        """
        dataframe_correlation_matrix: prints out the correlation matrix of the dataframe

        """
        return self.df.corr()
    
    def iqr(self, column_name):
        """
        iqr: returns the interquartile range

        """
        Q1 = self.df[column_name].quantile(0.25)
        Q3 = self.df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        return IQR





class DataFrameTransform:
    """
    This class is used for performing EDA transformations on the data.
    """
    def __init__(self, df):
        self.df = df
        
    def remove_columns(self):
        """
        drops columns based on inspections for skewness and outliers
        """
        drop_columns = ['mths_since_last_delinq', 'mths_since_last_record', 'next_payment_date', 'mths_since_last_major_derog', 'application_type', 'policy_code', 'out_prncp_inv', 'total_payment_inv', 'id']
        return self.df.drop(columns = drop_columns, inplace = True)
    
    def impute_median(self, name):
        """
        imputing method using a median
        """
        median_val = self.df[name].median() # Finds the median value of the column
        return self.df[name].fillna(value = median_val, inplace = True)
            
    def impute_mode(self, name):
        """
        imputing method using mode for objects and category data types
        """
        mode_val = self.df[name].mode().iloc[0] # Finds the mode value of the column
        return self.df[name].fillna(value = mode_val, inplace = True)
            
    def impute_null_values(self):
        """
        Main method for imputing null values, uses the previous two helper methods to impute null values
        """
        self.remove_columns()
        for name in self.df.columns:
            if self.df.dtypes[name] == 'int64' or self.df.dtypes[name] == 'float64' or self.df.dtypes[name] == 'datetime64[ns]':
                self.impute_median(name)
            elif self.df.dtypes[name] == 'object' or self.df.dtypes[name] == 'category':
                self.impute_mode(name)
               
    def skew_log_transform(self, column_names):
        """
        Method for transforming the data using log transform to correct skewness
        """
        for name in column_names:
            log_df_column = self.df[name].map(lambda x: np.log(x) if x > 0 else 0)
            # print(self.df[name])
            self.df[name] = log_df_column
            # print(self.df[name])
                
    def skew_boxcox_transform(self, column_names):
        """
        Method for transforming the data using boxcox transform to correct skewness
        """
        for name in column_names:      
            boxcox_column = self.df[name] + 0.01
            boxcox_column = stats.boxcox(boxcox_column)
            boxcox_column = pd.Series(boxcox_column[0])
            # print(self.df[name])
            self.df[name] = boxcox_column
            # print(self.df[name])
            
    def remove_IQR_outliers(self, column_names):
        """
        Method for removing outliers based on values from discovering the IQR
        """
        # Calculate the Q1, Q3, and IQR
        Q1 = self.df[column_names].quantile(0.25)
        Q3 = self.df[column_names].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Adjust lower bounds to be non-negative for count data
        lower_bound[lower_bound < 0] = 0
        
        # Remove outliers
        for name in column_names:
            self.df = self.df[(lower_bound[name] <= self.df[name]) & (self.df[name] <= upper_bound[name])]
    
    
#list of columns for boxcox transformation
boxcox_col = ['loan_amount', 'funded_amount', 'funded_amount_inv', 'int_rate', 'instalment', 'dti', 'total_payment']
#list of columns for log transformation
log_col = ['annual_inc', 'open_accounts', 'last_payment_amount']

    
trans_dataframe = DataFrameTransform(script_df)
trans_dataframe.impute_null_values()
trans_dataframe.skew_boxcox_transform(boxcox_col)
trans_dataframe.skew_log_transform(log_col)

# List of columns that are numerical and show skewness.
skew_num_cols = ['loan_amount', 'funded_amount', 'funded_amount_inv', 'int_rate', 'instalment', 'annual_inc','dti', 'total_payment', 'open_accounts','total_accounts', 'last_payment_amount']


class Plotter:
    """
    This class is used to visualise insights from the data.
    """
    def __init__(self, df):
        self.df = df
    
    def plot_missing_values(self):
        """
        This method plots missing values within the dataframe using a bar chart
        """
        msno.bar(self.df)
        plt.show()
        
    def null_matrix(self):
        """
        Plots missing values within the dataframe using a matrix
        """
        msno.matrix(self.df)
        plt.show()
    
    def plot_skew_data_hist(self, column_names):
        """
        Histogram used to plot and discover skewed data    
        """
        sns.set(font_scale=0.7)
        f = pd.melt(self.df, value_vars = column_names)
        g = sns.FacetGrid(f, col = "variable", col_wrap = 4, sharex = False, sharey = False)
        g = g.map(sns.histplot, "value", kde = True)
        plt.show()
        
    def plot_qq_skew(self, column_name):
        """
        QQ graph used to plot and discover skewed data    
        """
        self.df.sort_values(by = column_name, ascending=True)
        qq_plot = qqplot(self.df[column_name], scale=1, line='q')
        pyplot.show()
        
    def plot_skew_value(self, column_names):
        """
        A method for displaying the skew value of a column    
        """
        skew_value = []
        for name in column_names:
            skew_value.append([name, self.df[name].skew()])
            
        print(tabulate(skew_value, headers = ["Column", "Skewness"], tablefmt = "github"))
        
    def box_plot_outliers(self, column_names):
        """
        A box plot graph to plot and discover skewed data   
        """
        sns.set(font_scale = 0.7)
        f = pd.melt(self.df, value_vars = column_names)
        g = sns.FacetGrid(f, col = "variable", col_wrap = 4, sharex = False, sharey = False)
        g = g.map(sns.boxplot, "value", orient = "v", boxprops = dict(alpha=.3))
        plt.show()
        
    def visualize_correlation_matrix(self, column_names):
        """
        A method to display the correlation matrix of the data frame to discover correlation
        """
        correlation_matrix = self.df[column_names].corr()
        plt.figure(figsize = (14, 10))
        g = sns.heatmap(correlation_matrix, annot = True, fmt = '.1f', linewidth = .5)
        plt.show()

plot = Plotter(script_df)
# plot.plot_missing_values()
# plot.plot_skew_data_hist(skew_num_cols)
# plot.plot_qq_skew(skew_num_cols)
# plot.plot_skew_value(skew_num_cols)
# All the numerical data has a high positive skew meaning that the values are greatly over represented
# by outliers, also the mean and median are much higher than the mode

# plot.plot_skew_value(skew_num_cols)
# plot.plot_skew_data_hist(skew_num_cols)

# plot.box_plot_outliers(skew_num_cols)
# plot.visualize_correlation_matrix(skew_num_cols)




"""
The following columns will be removed due to high correlation:
- funded_amount
- funded_amount_inv
- instalment
- total_payment
"""
val_drop_columns = ['funded_amount', 'funded_amount_inv', 'instalment', 'total_payment']
script_df.drop(columns = val_drop_columns, inplace = True)


# final validated df
validated_df = script_df.copy()
# print(validated_df.dtypes)



def current_state_of_loans(df):
    
    current_state = df[['loan_amount', 'funded_amount_inv', 'total_payment']].copy()
    
    recovered_loan_percent = round((df['total_payment'] / df['loan_amount']) * 100, 2)
    funded_loan_percent = round((df['total_payment']/df['funded_amount_inv']) * 100, 2)
    
    current_state['recovered_against_investor (in %)'] = funded_loan_percent
    current_state['recovered_against_loan (in %)'] = recovered_loan_percent
    print(current_state)
    loan = df['loan_amount'].sum()
    investor = df['funded_amount_inv'].sum()
    payment = df['total_payment'].sum() 
    print(f'Recovered against loan: {round((payment/loan) * 100, 2)}%')
    print(f'Recovered against investor funded: {round((payment/investor) * 100, 2)}%')


# current_state_of_loans(original_df)

def calculate_loss(df):
    loss_df = df[['total_rec_prncp', 'loan_status', 'funded_amount']].copy()
    charged_off_sum = loss_df.loc[loss_df['loan_status'] == "Charged Off", 'total_rec_prncp'].sum()
    
    charged_off_counter = loss_df.loc[loss_df['loan_status'] == 'Charged Off', 'loan_status'].count()
    charged_off_percent = round(charged_off_counter / len(loss_df['loan_status']) * 100 , 2)
 
    print(f'Total paid for charged off loans: £{charged_off_sum}')
    print(f'Total percentage of charged off loans: {charged_off_percent}%')
    

# calculate_loss(original_df)

def calculate_projected_loss(df):
    df = df[['loan_amount', 'loan_status', 'term', 'int_rate', 'total_payment']].copy()
    
    df['term (in years)'] = df['term'].map(lambda x: 5 if x == '60 months' else 3) # Creates new column converting term into numeric years
    df['loan_inc_interest'] = round(df['loan_amount'] * (1 + (df['int_rate'] / 100)) ** df['term (in years)'], 2) # Creates new column that shows total loan
    
    loan_status_condition = df['loan_status'] == 'Charged Off'
    
    total_charged_off = df.loc[loan_status_condition]['loan_inc_interest'].sum() 
    total_charged_off_payment = round(df.loc[loan_status_condition]['total_payment'].sum(), 2)
    lost_revenue = round(total_charged_off - total_charged_off_payment, 2)
    
    total_revenue = df['loan_inc_interest'].sum()
    lost_revenue_percent = round((lost_revenue / total_revenue * 100), 2)
    
    print(f'Revenue Lost (£): {lost_revenue}')
    print(f'Revenue Lost (%): {lost_revenue_percent}%')

# calculate_projected_loss(original_df)
    

def possible_loss(df):
    df = df[['total_rec_prncp', 'loan_amount', 'loan_status', 'funded_amount', 'last_payment_amount', 'term', 'issue_date', 'last_payment_date', 'int_rate', 'total_payment']].copy()
    
    df['term (in years)'] = df['term'].map(lambda x: 5 if x == '60 months' else 3)
    df['loan_inc_interest'] = round(df['loan_amount'] * (1 + (df['int_rate'] / 100)) ** df['term (in years)'], 2)
    
    # condition for late
    condition_a = df['loan_status'].isin(['Late (16-30 days)', 'Late (31-120 days)'])
    
    total_late_loans = df.loc[condition_a]['loan_status'].count()
    late_loans_paid = df.loc[condition_a]['total_payment'].sum()
    late_loans_sum = df.loc[condition_a]['loan_inc_interest'].sum()
    late_loans_loss = round(late_loans_sum - late_loans_paid, 2)
    percentage_of_late_loans = round(total_late_loans / (df["loan_status"]).count() * 100, 2)
    
    # condition inc. late, default, charged off
    condition_b = df['loan_status'].isin(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'Charged Off'])
    
    total_lost_loan = df.loc[condition_b]['loan_inc_interest'].sum() 
    total_lost_paid = df.loc[condition_b]['total_payment'].sum() 
    total_lost_value = round(total_lost_loan - total_lost_paid, 2)
    percentage_lost_total_rev = round(total_lost_value / df["loan_inc_interest"].sum() * 100, 2)

    print(f'Total late status: {total_late_loans}')
    print(f'% of members in late Status: {percentage_of_late_loans}%')
    print(f'Total lost if late loans Charged Off: £{late_loans_loss}')
    print(f'Total lost for late, default and Charged Off loans: £{total_lost_value}')
    print(f'% Lost of Total Revenue: {percentage_lost_total_rev}%')
    
    
# possible_loss(original_df)

def loss_indicators(df):
    # list all potential indicators to loss
    potential_indicators = ['loan_status','purpose', 'sub_grade', 'dti', 'employment_length', 'annual_inc', 'verification_status', 'delinq_2yrs', 'int_rate', 'out_prncp', 'inq_last_6mths', 'loan_amount', 'open_accounts', 'total_accounts']
    indicator_df = df[potential_indicators].copy() # Create new dataframe
    
    loan_status_dict = {
                        'Charged Off' : 9,
                        'Default' : 8,
                        'Late (31-120 days)' : 5,
                        'Late (16-30 days)' : 6,    
                        'In Grace Period' : 0,
                        'Does not meet the credit policy. Status:Charged Off' : 0,
                        'Does not meet the credit policy. Status:Fully Paid' : 0,
                        'Current' : 0,                                                                                                                                                                      
                        'Fully Paid' : 4                                               
                    }
    indicator_df.replace({'loan_status' : loan_status_dict}, inplace = True)
    
    for indicator in potential_indicators:
        if type(indicator_df[indicator][0]) != int or type(indicator_df[indicator][0]) != float:
            code, unique = pd.factorize(indicator_df[indicator], sort = True)
            indicator_df[indicator] = code
    
    condition_a = df['loan_status'].isin(['Late (31-120 days)', 'Late (16-30 days)', 'Charged Off', 'Default', 'Fully Paid'])
    indicator_df = indicator_df.loc[condition_a]
    
    plt.figure(figsize = (14, 10))
    g = sns.heatmap(indicator_df.corr(), annot = True, fmt = '.1f', linewidth = .5)
    plt.show()


# loss_indicators(original_df)
