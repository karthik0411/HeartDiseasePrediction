from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
def preprocessing(dataset, y):
    target_col = [y]
    cat_cols   = dataset.nunique()[dataset.nunique() < 5].keys().tolist()
    cat_cols   = [x for x in cat_cols ]
    #numerical columns
    num_cols   = [x for x in dataset.columns if x not in cat_cols + target_col]
    #Binary columns with 2 values
    bin_cols   = dataset.nunique()[dataset.nunique() == 2].keys().tolist()
    #Columns more than 2 values
    multi_cols = [i for i in cat_cols if i not in bin_cols]

    #Label encoding Binary columns
    le = LabelEncoder()
    for i in bin_cols :
        dataset[i] = le.fit_transform(dataset[i])

    #Duplicating columns for multi value columns
    dataset = pd.get_dummies(data = dataset,columns = multi_cols )

    #Scaling Numerical columns
    std = StandardScaler()
    scaled = std.fit_transform(dataset[num_cols])
    scaled = pd.DataFrame(scaled,columns=num_cols)

    #dropping original values merging scaled values for numerical columns
    df_dataset_og = dataset.copy()
    dataset = dataset.drop(columns = num_cols,axis = 1)
    dataset = dataset.merge(scaled,left_index=True,right_index=True,how = "left")
    return dataset
def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers


df = pd.read_csv('heart.csv')
#data =preprocessing(data, 'sex')
#df['sex']=df['sex'].astype('category')
print(df.head())
Outliers_to_drop = detect_outliers(df,2,['trestbps', 'chol','thalach'])
print(df.loc[Outliers_to_drop])
def makeCategorical(dataset,features=list()):
    for f in features:
        dataset[f]= dataset[f].astype('category')
    return dataset
features=['sex','cp','fbs','restecg','exang','slope','ca','thal','target']
df = makeCategorical(df,features)
print(df.dtypes)
y=df['target']
df=pd.get_dummies(df,drop_first=True)
print(df.head())
df.to_csv('test1.csv')
X=df.drop('target_1',axis=1)
print(X.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

classifiers=[['Logistic Regression :',LogisticRegression(max_iter=200,solver='liblinear')],
       ['Decision Tree Classification :',DecisionTreeClassifier()],
       ['Random Forest Classification :',RandomForestClassifier(n_estimators=100)],
       ['Gradient Boosting Classification :', GradientBoostingClassifier()],
       ['Ada Boosting Classification :',AdaBoostClassifier()],
       ['Extra Tree Classification :', ExtraTreesClassifier(n_estimators=100)],
       ['K-Neighbors Classification :',KNeighborsClassifier()],
       ['Support Vector Classification :',SVC(gamma='scale')],
       ['Gaussian Naive Bayes :',GaussianNB()]]
cla_pred=[]
for name,model in classifiers:
    model=model
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    cla_pred.append(accuracy_score(y_test,predictions))
    print(name,accuracy_score(y_test,predictions))
'''
sns.set_style("darkgrid")
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu',fmt='.2f',linewidths=2)
plt.show()
print(df['target'].value_counts())
sns.distplot(df['age'],color='Red',hist_kws={'alpha':1,"linewidth": 2}, kde_kws={"color": "k", "lw": 3, "label": "KDE"})
plt.show()

fig,ax=plt.subplots(figsize=(24,6))
plt.subplot(1, 3, 1)
age_bins = [20,30,40,50,60,70,80]
df['bin_age']=pd.cut(df['age'], bins=age_bins)
g1=sns.countplot(x='bin_age',data=df ,hue='target',palette='plasma',linewidth=3)
g1.set_title("Age vs Heart Disease")
#The number of people with heart disease are more from the age 41-55
#Also most of the people fear heart disease and go for a checkup from age 55-65 and dont have heart disease (Precautions)

plt.subplot(1, 3, 2)
cho_bins = [100,150,200,250,300,350,400,450]
df['bin_chol']=pd.cut(df['chol'], bins=cho_bins)
g2=sns.countplot(x='bin_chol',data=df,hue='target',palette='plasma',linewidth=3)
g2.set_title("Cholestoral vs Heart Disease")
#Most people get the heart disease with 200-250 cholestrol 
#The others with cholestrol of above 250 tend to think they have heart disease but the rate of heart disease falls

plt.subplot(1, 3, 3)
thal_bins = [60,80,100,120,140,160,180,200,220]
df['bin_thal']=pd.cut(df['thalach'], bins=thal_bins)
g3=sns.countplot(x='bin_thal',data=df,hue='target',palette='plasma',linewidth=3)
g3.set_title("Thal vs Heart Disease")

plt.show()
fig,ax=plt.subplots(figsize=(24,6))
plt.subplot(131)
x1=sns.countplot(x='cp',data=df,hue='target',palette='spring',linewidth=3)
x1.set_title('Chest pain type')
#Chest pain type 2 people have highest chance of heart disease

plt.subplot(132)
x2=sns.countplot(x='thal',data=df,hue='target',palette='spring',linewidth=3)
x2.set_title('Thal')
#People with thal 2 have the highest chance of heart disease

plt.subplot(133)
x3=sns.countplot(x='slope',data=df,hue='target',palette='spring',linewidth=3)
x3.set_title('slope of the peak exercise ST segment')
plt.show()
'''
