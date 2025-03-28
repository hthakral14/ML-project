import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
df=pd.read_csv('notebook/HR_comma_sep.csv', encoding = 'utf-8')

def reading_cleaning(df):
    sal={
        'high':3,
        'medium':2,
        'low':1,
        'h':3,
    }

    df['num_salary']=df.salary.apply(lambda x: sal[x])
    cols=df.columns.tolist()
    cols.insert(len(cols),cols.pop(cols.index('left')))
    df=df[cols]
    df.columns=[x.lower() for x in cols]
    df.drop_duplicates(inplace=True)
    
    return df
df = reading_cleaning(df)
df.head()
def employee_important_info(df):
    # Average satisfaction level
    average_satisfaction = df['satisfaction_level'].mean()
    # Department-wise average satisfaction level
    department_satisfaction = df.groupby('department')['satisfaction_level'].mean()
    # Salary-wise average satisfaction level
    salary_satisfaction = df.groupby('salary')['satisfaction_level'].mean()

    # Employees who left
    left_employees = len(df[df['left'] == 1])
    # Employees who stayed
    stayed_employees = len(df[df['left'] == 0])
    
    return average_satisfaction,department_satisfaction,salary_satisfaction,left_employees,stayed_employees

average_satisfaction,department_satisfaction,salary_satisfaction,left_employees,stayed_employees = employee_important_info(df)
print("Average Satisfaction Level:", average_satisfaction)
print("Department-wise Average Satisfaction Level:\n", department_satisfaction)
print("Salary-wise Average Satisfaction Level:\n", salary_satisfaction)
print("Employees who left:\n", left_employees)
print("Employees who stayed:\n", stayed_employees)
def plots(df,col):
    values = df[col].unique()
    plt.figure(figsize=(15,8))
    
    explode = [0.1 if len(values) >1 else 0] * len(values)
    plt.pie(df[col].value_counts(),explode=explode,startangle=40,autopct='%1.1f%%',shadow=True)
    labels = [f'{value} ({col})' for value in values]
    plt.legend(labels=labels,loc='upper right')
    
    plt.title(f"distribution of {col}")
    plt.show()
    
plots(df,'left')
plots(df,'salary')
plots(df,'number_project')
plots(df,'department')
def distribution(df,col):
    values = df[col].unique()
    plt.figure(figsize=(15,8))
    sns.countplot(x=df[col],hue='left',palette='Set1',data=df)
    labels = [f"{val} ({col})" for val in values]
    plt.legend(labels=labels,loc="upper right")
    plt.title(f"distribution of {col}")
    plt.xticks(rotation=90)
    plt.show()
distribution(df,'salary')
distribution(df,'department')
def comparison(df,x,y):
    plt.figure(figsize=(15,8))
    sns.barplot(x=x,y=y,hue='left',data=df,ci=None)
    plt.title(f'{x} vs {y}')
    plt.show()
comparison(df,'department', 'satisfaction_level')
def corr_with_left(df):
    df_encoded = pd.get_dummies(df)
    correlations = df_encoded.corr()['left'].sort_values()[:-1]
    colors = ['skyblue' if corr>=0 else 'salmon' for corr in correlations]
    plt.figure(figsize=(10,8))
    correlations.plot(kind='barh', color=colors)
    # Add title and labels
    plt.title('Correlation with Left')
    plt.xlabel('Correlation')
    plt.ylabel('Features')

    # Show the plot
    plt.show()
corr_with_left(df)
def histogram(df, col):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Create a grid of 1 row and 2 columns

    # Plot the first histogram
    sns.histplot(data=df, x=col, hue='left', bins=20, ax=axes[0])
    axes[0].set_title(f"Histogram of {col}")

    # Plot the second histogram
    sns.kdeplot(data=df, x='satisfaction_level', y='last_evaluation', hue='left', shade=True, ax=axes[1])
    axes[1].set_title("Kernel Density Estimation")

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()

histogram(df, 'satisfaction_level')
from sklearn.cluster import KMeans
def leftKmeans(df):
    df1=df[df.left==1].copy()
    df1 = pd.get_dummies(df1)
    kmeans = KMeans(n_clusters=3,random_state=2)
    df1['kmean_label']=kmeans.fit_predict(df1[["satisfaction_level","last_evaluation"]])

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(x='satisfaction_level',y='last_evaluation',data=df1,
                    hue='kmean_label',palette=['g','r','b'],alpha=0.8)
    plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],
                color="black",marker="X",s=100)
    plt.xlabel("satisfaction")
    plt.ylabel("Evaluation")
    plt.title("Clusters of Employee churn")
    plt.show()
    
    return df1


df1 = pd.DataFrame()
df1 = leftKmeans(df)
df1.groupby('kmean_label').mean()