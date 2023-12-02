# Exploratory Data Analysis - Customer Loans in Finance

## Project Description
This project focuses on managing loans and make informed decisions about loan approvals and manage risks efficiently, by performing exploratory data analysis on the loan portfolio, using a variety of statistical and visualisation techniques to discover patterns, relationships and anomalies in the loan portfolio.

## Installation instructions
1. **Clone this repository**: Clone this repository onto your local machine to get the necessary files and tools to get started.
2. **Ensure tools and libraries are installed**: This makes use of Python 3 and uses the following libraries:
- Pandas
- SQLAlchemy
- PyYAML
Ensure all necessary libraries and tools and installed.
3. **Execute the `db_utils.py` file**: This ensures that the database is connected and saves it locally as a csv file.

## Milestones
The following shows the milestones that were followed when completing the project.

- **Milestone 1**: Setting up the environment
    - Created a new GitHub repository, then cloned onto local machine
    - Installed necessary tools/packages/libraries to start making developments

- **Milestone 2**: Extract the loans data from the cloud
    - Created the `db_utils.py` file which contains the `RDSDatabaseConnector` class to extract the data from the database.
    - Created the `credentials.yaml` file which is used to store the database credentials and added the yaml file to the `.gitignore` file for security reasons
    - Defined the `RDSDatabaseConnector` class to take the `credentials.yaml` file and return the data dictionary as the class will use to connect to the remote database.
    - Defined the initialise method to initialise the dictionary and the SQLAlchemy engine.
    - Defined a separate initialise method for the SQLAlchemy engine to be used in the main initilising method.
    - Defined a method to extract the credentials provided by the class using the Pandas library.
    - Defined a method to store the extracted data locally as a `loan_payments.csv` file. 
- **Milestone 3**: Exploratory Data Analysis
    - Converted the columns to the correct format
    - Created a `DataFrameInfo` class which contained methods to extract information from the dataframe.
    - Created a `Plotter` class for visualisation and insights of the data
    - Created methods within the two classes to discover skewness, outlier and correlation between data in the dataframe to judge which columns to remove from analysis
-**Milestone 4**: Analysis and Visualation:
    - Focused on further analysis of the data to get deeper insights to find patterns or trends.
    - Used the data to discover areas like: Calculating loss, calculating projected loss, possible loss and indicators of loss

## File structure
- **`credentials.yaml`**: A hidden file which contains the database credentials, added to the `.gitignore` file.
- **`db_utils.py`**: Python file which contains the class to extract and download the database.

## License information

