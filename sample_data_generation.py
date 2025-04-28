import pandas as pd
#Creates sample data csv file by taking random points(equivalent to sample size) from complete data file
def create_sample_data(complete_data_file, sample_data_file, sample_size):
    complete_data = pd.read_csv(complete_data_file, header=None)
    sample_data = complete_data.sample(n=sample_size, random_state=1)
    sample_data.to_csv(sample_data_file, index=False, header=False)

#Usage in CURE
complete_data_file = 'complete_age_bmi_data.csv'
sample_data_file = 'sample_age_bmi_data.csv'
sample_size = 100 
create_sample_data(complete_data_file, sample_data_file, sample_size)
