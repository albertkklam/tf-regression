# Set up the work directory
# Here, our complete data set is in a tab delimited file called data_v.txt

data_dir = '/path/to/directory/'
data_file = data_dir + 'data_v.txt'

# Specify the data types here before parsing
# We ensure that our response here RESPONSE is a float

dtypes = {'RESPONSE': np.float}

# Our file is tab delimited, so we use the tab separator here
# We also have a date column called WEEK in DD-MM-YYYY format, so we need to parse this in as a date format
# to_datetime is a clever pandas function that can detect the date format
# You may need to do some fiddling with the parameters of to_datetime if it initially fails to detect the date format

rawdata = pd.read_csv(data_file, sep = "\t", dtype = dtypes, parse_dates = ['WEEK'], date_parser = pd.to_datetime)

# Define a simple function to remove spaces and other punctuation
# We will eventually make our categorical variables into dummy variables
# Our categorical values will form the basis for our dummy variable column names

def df_strreplace(df):
    df.replace(r'[\s]','_', inplace = True, regex = True)
    df.replace(r'[\.]','', inplace = True, regex = True)
    df.replace(r'__','_', inplace = True, regex = True)

df_strreplace(rawdata)

# Define a simple function to create dummy variables for our categorical variables cat_var_list
# We append the categorical value to the end of the categorical column name for each dummy

def df_cat(df, cat_var_list):
    for cv in cat_var_list:
        if [i for i, x in enumerate(cat_var_list) if cv == x][0] == 0:
            dummy_df = pd.get_dummies(df[cv], prefix = cv)
        else:
            dummy_df = pd.concat([dummy_df, pd.get_dummies(df[cv], prefix = cv)], axis = 1)
    return dummy_df

# Define cat_var_list and other columns here
# cat_cols holds our categorical columns
# fac_cols holds columns that we will not use for training or testing
# pred_cols holds our response columns
# num_cols holds our numeric columns
# We append on the WEEK column at the end of num_cols as we will use this to separate our train and test sets

cat_cols = ['ATTRIBUTE_1','ATTRIBUTE_2','ATTRIBUTE_3']
fac_cols = ['FACTOR_1','FACTOR_2','FACTOR_3']
pred_cols = ['RESPONSE']
num_cols = [x for x in list(rawdata.columns.values) if x not in cat_cols if x not in fac_cols if x not in pred_cols]
num_cols.append('WEEK')

# Subset data based on our columns above
# Apply df_cat to our categorical columns here

num_rawdata = rawdata[num_cols]
cat_rawdata = df_cat(rawdata[cat_cols], cat_cols)
pred_rawdata = rawdata[pred_cols]

# Stack our categorical and numeric data sets together
# Define our data_split variable that will separate our train and test set
# Split x_rawdata into train and test based on date_split and remove the WEEK column
# Note that hstack converts our pandas dataframe into a numpy array

x_rawdata = np.hstack((cat_rawdata, num_rawdata))
date_split = dt.datetime(2016,1,1,0,0,0)
x_train = x_rawdata[x_rawdata[:,-1] <= date_split][:,:-1]
x_test = x_rawdata[x_rawdata[:,-1] > date_split][:,:-1]

# Convert our response dataframe into a numpy array and split train and test similarly

y_rawdata = pred_rawdata.as_matrix() 
y_train = y_rawdata[y_rawdata[:,-2] <= date_split][:,-1]
y_test = y_rawdata[y_rawdata[:,-2] > date_split][:,-1]

# We now have our four arrays (X,y for train and test) that are ready to feed into our TensorFlow pipeline
