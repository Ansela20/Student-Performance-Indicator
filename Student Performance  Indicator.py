#!/usr/bin/env python
# coding: utf-8

# ## **Student Performance Indicator**
# 
# **Life Cycle of Machine Learning Project**
# 
# - Understanding the Problem Statement
# - Data Collection
# -Data Checks to perform
# -Exploratory Data Analysis
# -Data Pre-Processing
# -Model Training
# -Choose Best Model

# ### **1) Problem statement**
# 
# This project understands how the Student's Performance (test scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Meal and Test preparation course.

# ### **2) Data Collection**
# 
# - Dataset Source - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977
# 
# - The data consists of 8 column and 1000 rows.

# ### **2.1 Import Data and Required Packages**
# 
# **Importing Pandas, Numpy, Matplotlib, Seaborn and Warings Library.**

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# **Import the CSV Data as Pandas DataFrame**

# In[2]:


df = pd.read_csv('C:/Users/Ansela/Downloads/student_data.csv')


# **Show Top 10 Records**

# In[76]:


df.head(10)


# **Shape of the dataset**

# In[4]:


df.shape


# ### **2.2 Dataset information**
# 
# - gender : sex of students -> (male/female)
# - race_ethnicity : ethnicity of students -> (Group A, B,C, D,E)
# - parental_level_of_education : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,high school)
# - meal : having meal before test (standard or free/reduced)
# - test_preparation_course : complete or not complete before test
# - math_score
# - reading_score
# - writing_score

# ### **3. Data Checks to perform**
# 
# - Check missing values
# - Check duplicates
# - Check data type
# - Check the number of unique values of each column
# - Check statistics of data set
# - Check various categories present in the different categorical column

# ### **3.1 Check missing values**

# In[5]:


df.isna().sum()


# ### **3.2 Check duplicates**

# In[6]:


df.duplicated().sum()


# **There are no duplicates values in the data**

# ### **3.3 Check data types**

# In[7]:


# Check Null and Dtypes
df.info()


# ### **3.4 Checking the number of unique values of each column**

# In[8]:


df.nunique()


# ### **3.5 Check statistics of data set**

# In[9]:


df.describe()


# **Insight**
# - From above description of numerical data, all means are very close to each other - between 66 and 68.05;
# - All standard deviations are also close - between 14.6 and 15.19;
# - While there is a minimum score 0 for math, for writing minimum is much higher = 10 and for reading minimum score = 17

# ### **3.7 Exploring Data**

# In[10]:


df.head()


# In[11]:


# Explore the categories in variable
print("Categories in 'gender' variable: ",end=" " )
print(df['gender'].unique())

print("Categories in 'race_ethnicity' variable: ",end=" ")
print(df['race_ethnicity'].unique())

print("Categories in'parental_level_of_education' variable: ",end=" " )
print(df['parental_level_of_education'].unique())

print("Categories in 'meal' variable: ",end=" " )
print(df['meal'].unique())

print("Categories in 'test_preparation_course' variable: ",end=" " )
print(df['test_preparation_course'].unique())


# In[12]:


# define numerical & categorical columns
numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))


# In[13]:


df.head(2)


# ### **3.8 Adding columns for "Total Score" and "Average"**

# In[14]:


df['total score'] = df['math_score'] + df['reading_score'] + df['writing_score']
df['average'] = df['total score']/3

df.head()


# In[15]:


df = df.rename(columns={'total score': 'total_score'})

df.head()


# In[16]:


# Number of students with full marks

reading_full = df[df['reading_score'] == 100]['average'].count()
writing_full = df[df['writing_score'] == 100]['average'].count()
math_full = df[df['math_score'] == 100]['average'].count()

print(f'Number of students with full marks in Maths: {math_full}')
print(f'Number of students with full marks in Writing: {writing_full}')
print(f'Number of students with full marks in Reading: {reading_full}')


# In[21]:


# Number of students with less than 20 marks

reading_less_20 = df[df['reading_score'] <= 20]['average'].count()
writing_less_20 = df[df['writing_score'] <= 20]['average'].count()
math_less_20 = df[df['math_score'] <= 20]['average'].count()

print(f'Number of students with less than 20 marks in Maths: {math_less_20}')
print(f'Number of students with less than 20 marks in Writing: {writing_less_20}')
print(f'Number of students with less than 20 marks in Reading: {reading_less_20}')


# **Insights**
# - From above values, we see that performance of students is better in reading but maths
# 

# ### **4. Exploring Data ( Visualization )**

# ### **4.1 Visualize average score distribution to draw some conclusion.**
# - Histogram
# - Kernel Distribution Function (KDE)

# ### **4.1.1 Histogram & KDE**

# In[124]:


# Average score distribution of students
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df,x='average',bins=30,kde=True,color='g')
plt.subplot(122)
sns.histplot(data=df,x='average',kde=True,hue='gender')
plt.show()


# **Insights**
# - Female students tend to perform well then male students.

# In[120]:


plt.subplots(1,3,figsize=(25,6))
plt.subplot(141)
sns.histplot(data=df,x='average',kde=True,hue='meal')
plt.subplot(142)
sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='meal')
plt.subplot(143)
sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='meal')
plt.show()


# **Insights**
# - Standard meal helps perform well in exams.
# - Standard meal helps perform well in exams be it a male or a female.

# In[30]:


plt.subplots(1,3,figsize=(25,6))
plt.subplot(141)
ax =sns.histplot(data=df,x='average',kde=True,hue='race_ethnicity')
plt.subplot(142)
ax =sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='race_ethnicity')
plt.subplot(143)
ax =sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='race_ethnicity')
plt.show()


# **Insights**
# - Students of group A and group B tends to perform poorly in exam.
# - Students of group A and group B tends to perform poorly in exam irrespective of whether they are male or female

# ### **4.2 Maximumum score of students in all three subjects**

# In[122]:


plt.figure(figsize=(18,8))
plt.subplot(1, 4, 1)
plt.title('MATH SCORES')
sns.violinplot(y='math_score',data=df,color='red',linewidth=3)
plt.subplot(1, 4, 2)
plt.title('READING SCORES')
sns.violinplot(y='reading_score',data=df,color='green',linewidth=3)
plt.subplot(1, 4, 3)
plt.title('WRITING SCORES')
sns.violinplot(y='writing_score',data=df,color='blue',linewidth=3)
plt.show()


# **Insights**
# - From the above three plots its clearly visible that most of the students score in between 60-80 in Maths whereas in reading and writing most of them score from 50-80

# ### **4.3 Multivariate analysis using pieplot**

# In[116]:


# Set the figure size
plt.rcParams['figure.figsize'] = (30, 25)

# Create the first pie chart
plt.subplot(2, 3, 1)
size = df['gender'].value_counts()
labels = ['Female', 'Male']
colors = ['red', 'green']
plt.pie(size, colors=colors, labels=labels, autopct='%1.2f%%', textprops={'fontsize': 24})
plt.title('Gender', fontsize=28)
plt.axis('off')

# Create the second pie chart
plt.subplot(2, 3, 2)
size = df['race_ethnicity'].value_counts()
labels = ['Group C', 'Group D', 'Group B', 'Group E', 'Group A']
colors = ['red', 'green', 'blue', 'cyan', 'orange']
plt.pie(size, colors=colors, labels=labels, autopct='%1.2f%%', textprops={'fontsize': 24})
plt.title('Race_Ethnicity', fontsize=28)
plt.axis('off')

# Create the third pie chart
plt.subplot(2, 3, 3)
size = df['meal'].value_counts()
labels = ['Standard', 'Free']
colors = ['red', 'green']
plt.pie(size, colors=colors, labels=labels, autopct='%1.2f%%', textprops={'fontsize': 24})
plt.title('Meal', fontsize=28)
plt.axis('off')

# Create the fourth pie chart
plt.subplot(2, 3, 4)
size = df['test_preparation_course'].value_counts()
labels = ['None', 'Completed']
colors = ['red', 'green']
plt.pie(size, colors=colors, labels=labels, autopct='%1.2f%%', textprops={'fontsize': 24})
plt.title('Test Course', fontsize=28)
plt.axis('off')

# Create the fifth pie chart
plt.subplot(2, 3, 5)
size = df['parental_level_of_education'].value_counts()
labels = ['Some College', "Associate's Degree", 'High School', 'Some High School', "Bachelor's Degree", "Master's Degree"]
colors = ['red', 'green', 'blue', 'cyan', 'orange', 'grey']
plt.pie(size, colors=colors, labels=labels, autopct='%1.2f%%', textprops={'fontsize': 24})
plt.title('Parental Education', fontsize=28)
plt.axis('off')

# Adjust layout
plt.tight_layout()

# Display the pie charts
plt.show()


# **Insights**
# - Number of Male and Female students are almost equal
# - Number students are greatest in Group C
# - Number of students who have standard meal are greater
# - Number of students who have not enrolled in any test preparation course are greater
# - Number of students whose parental education is "Some College" are greater followed by "Associate's Degree"

# ### **4.4 Feature Wise Visualization**
# 

# ### **BIVARIATE ANALYSIS ( Does gender has any impact on student's performance ? )**

# In[41]:


gender_group = df.groupby('gender').mean()
gender_group


# In[42]:


plt.figure(figsize=(10, 8))

X = ['Total Average','Math Average']


female_scores = [gender_group['average'][0], gender_group['math_score'][0]]
male_scores = [gender_group['average'][1], gender_group['math_score'][1]]

X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, male_scores, 0.4, label = 'Male')
plt.bar(X_axis + 0.2, female_scores, 0.4, label = 'Female')
  
plt.xticks(X_axis, X)
plt.ylabel("Marks")
plt.title("Total average v/s Math average marks of both the genders", fontweight='bold')
plt.legend()
plt.show()


# **Insights**
# - On an average females have a better overall score than men.
# - whereas males have scored higher in Maths.

# ### **4.4.2 RACE/EHNICITY COLUMN**
# - How is Group wise distribution ?
# - Is Race/Ehnicity has any impact on student's performance?

# ### **UNIVARIATE ANALYSIS ( How is Group wise distribution ?)**

# In[44]:


f,ax=plt.subplots(1,2,figsize=(20,10))
sns.countplot(x=df['race_ethnicity'],data=df,palette = 'bright',ax=ax[0],saturation=0.95)
for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=20)
    
plt.pie(x = df['race_ethnicity'].value_counts(),labels=df['race_ethnicity'].value_counts().index,explode=[0.1,0,0,0,0],autopct='%1.1f%%',shadow=True)
plt.show()   


# **Insights**
# - Most of the student belonging from group C and group D.
# - Lowest number of students belong to group A.

# ### **BIVARIATE ANALYSIS ( Is Race/Ehnicity has any impact on student's performance ? )**

# In[46]:


Group_data2=df.groupby('race_ethnicity')
f,ax=plt.subplots(1,3,figsize=(20,8))
sns.barplot(x=Group_data2['math_score'].mean().index,y=Group_data2['math_score'].mean().values,palette = 'mako',ax=ax[0])
ax[0].set_title('Math score',color='#005ce6',size=20)

for container in ax[0].containers:
    ax[0].bar_label(container,color='black',size=15)

sns.barplot(x=Group_data2['reading_score'].mean().index,y=Group_data2['reading_score'].mean().values,palette = 'flare',ax=ax[1])
ax[1].set_title('Reading score',color='#005ce6',size=20)

for container in ax[1].containers:
    ax[1].bar_label(container,color='black',size=15)

sns.barplot(x=Group_data2['writing_score'].mean().index,y=Group_data2['writing_score'].mean().values,palette = 'coolwarm',ax=ax[2])
ax[2].set_title('Writing score',color='#005ce6',size=20)

for container in ax[2].containers:
    ax[2].bar_label(container,color='black',size=15)


# **Insights**
# - Group E students have scored the highest marks.
# - Group A students have scored the lowest marks.
# - Students from a lower socioeconomic status have a lower avg in all course subjects

# ### **4.4.3 PARENTAL LEVEL OF EDUCATION COLUMN**
# 
# - What is educational background of student's parent ?
# - Is parental education has any impact on student's performance ?
# - UNIVARIATE ANALYSIS ( What is educational background of student's parent ? )

# In[48]:


plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')
sns.countplot(df['parental_level_of_education'], palette = 'Blues')
plt.title('Comparison of Parental Education', fontweight = 30, fontsize = 20)
plt.xlabel('Degree')
plt.ylabel('count')
plt.show()


# **Insights**
# - Largest number of parents are from some college.

# ### **BIVARIATE ANALYSIS ( Is parental education has any impact on student's performance?)**

# In[133]:


df.groupby('parental_level_of_education').agg('mean').plot(kind='barh',figsize=(10,10))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# **Insights**
# - The score of student whose parents possess master and bachelor level education are higher than others.

# ### **4.4.4 TEST PREPARATION COURSE COLUMN**
# - Is Test prepration course has any impact on student's performance ?

# ### **BIVARIATE ANALYSIS ( Is Test prepration course has any impact on student's performance ? )**

# In[137]:


plt.figure(figsize=(12,6))

# Subplot 1: Math Scores
plt.subplot(2, 2, 1)
sns.barplot(x=df['meal'], y=df['math_score'], hue=df['test_preparation_course'])
plt.title('Math Scores by Meal Type and Test Prep')
plt.legend(loc='upper right')

# Subplot 2: Reading Scores
plt.subplot(2, 2, 2)
sns.barplot(x=df['meal'], y=df['reading_score'], hue=df['test_preparation_course'])
plt.title('Reading Scores by Meal Type and Test Prep')
plt.legend(loc='upper right')

# Subplot 3: Writing Scores
plt.subplot(2, 2, 3)
sns.barplot(x=df['meal'], y=df['writing_score'], hue=df['test_preparation_course'])
plt.title('Writing Scores by Meal Type and Test Prep')
plt.legend(loc='upper right')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# ### **4.4.5 CHECKING OUTLIERS**

# In[64]:


plt.subplots(1,4,figsize=(16,5))
plt.subplot(141)
sns.boxplot(df['math_score'],color='skyblue')
plt.subplot(142)
sns.boxplot(df['reading_score'],color='hotpink')
plt.subplot(143)
sns.boxplot(df['writing_score'],color='yellow')
plt.subplot(144)
sns.boxplot(df['average'],color='lightgreen')
plt.show()


# ### **4.4.6 MUTIVARIATE ANALYSIS USING PAIRPLOT**

# In[123]:


sns.pairplot(df,hue = 'gender')
plt.show()


# **Insights**
# - From the above plot it is clear that all the scores increase linearly with each other.

# ### **5. Conclusion**
# 
# - Student's Performance is related with meal, race, parental level education
# - Females lead in pass percentage and also are top-scorers
# - Student's Performance is not much related with test preparation course
# - Finishing preparation course is benefitial.

# ### **Model Training**
# 
# ### **1.1 Import Data and Required Packages**
# Importing Pandas, Numpy, Matplotlib, Seaborn and Warings Library.

# In[74]:


# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import warnings


# **Show Top 5 Records**

# In[75]:


df.head()


# **Preparing X and Y variables**

# In[77]:


X = df.drop(columns=['total_score', 'average'], axis=1)


# In[78]:


X.head()


# In[80]:


Y = df['total_score']


# In[81]:


Y.head()


# In[87]:


print("Categories in 'gender' variable: ",end=" " )
print(df['gender'].unique())

print("Categories in 'race_ethnicity' variable: ",end=" ")
print(df['race_ethnicity'].unique())

print("Categories in'parental_level_of_education' variable: ",end=" " )
print(df['parental_level_of_education'].unique())

print("Categories in 'meal' variable: ",end=" " )
print(df['meal'].unique())

print("Categories in 'test_preparation_course' variable: ",end=" " )
print(df['test_preparation_course'].unique())


# In[82]:


# Create Column Transformer with 3 types of transformers
num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_features),
         ("StandardScaler", numeric_transformer, num_features),        
    ]
)


# In[83]:


X = preprocessor.fit_transform(X)


# In[85]:


X


# In[86]:


X.shape


# In[88]:


# separate dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
X_train.shape, X_test.shape


# **Develop a Function to Evaluate Model Performance Metrics Post-Training**

# In[92]:


def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


# In[93]:


models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "AdaBoost Regressor": AdaBoostRegressor()
}
model_list = []
r2_list =[]

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, Y_train) # Train model

    # Make predictions
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    
    # Evaluate Train and Test dataset
    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(Y_train, Y_train_pred)

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(Y_test, Y_test_pred)

    
    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')
    
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    r2_list.append(model_test_r2)
    
    print('='*35)
    print('\n')


# In[94]:


pd.DataFrame(list(zip(model_list, r2_list)), columns=['Model Name', 'R2_Score']).sort_values(by=["R2_Score"],ascending=False)


# **Linear Regression**

# In[96]:


lin_model = LinearRegression(fit_intercept=True)
lin_model = lin_model.fit(X_train, Y_train)
Y_pred = lin_model.predict(X_test)
score = r2_score(Y_test, Y_pred)*100
print(" Accuracy of the model is %.2f" %score)


# **Plot Y_pred and Y_test**

# In[97]:


plt.scatter(Y_test,Y_pred);
plt.xlabel('Actual');
plt.ylabel('Predicted');


# In[99]:


sns.regplot(x=Y_test,y=Y_pred,ci=None,color ='red');


# In[100]:


pred_df=pd.DataFrame({'Actual Value':Y_test,'Predicted Value':Y_pred,'Difference':Y_test - Y_pred})
pred_df


# In[ ]:




