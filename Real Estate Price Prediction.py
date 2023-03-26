import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)

df = pd.read_csv("Bengaluru_House_Data.csv")
# print(df)

# print(df.columns)

# print(df.groupby('area_type')['area_type'].agg('count'))                     #Study more about this
#Alternate to groupby + Agg(count) is .value_counts() check location_stat parameter in df4

df1 = df.drop(['availability', 'balcony','society','area_type'],axis ='columns')
# print(df1)

# print(df1.isnull().sum())

df2 = df1.dropna()
# print(df2)

# print(df2['size'].unique())

"To remove words like BHK, Bedroom, RK etc"
df2['bhk']=df2['size'].apply(lambda x : int(x.split(' ')[0]))                    #Use apply word to apply fn on a column
# print(df2)

# print(df2['total_sqft'].unique())

"To segregate range values like 1133 - 1384"

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# print(df2[~df2['total_sqft'].apply(is_float)])         # '~' This is negate ie throw out

"To convert the range value into a single value"

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df3 = df2.copy()
df3['total_sqft'] = df3['total_sqft'].apply(convert_sqft_to_num)
# print(df3)


df4 = df3.copy()
df4['price_per_sqft'] = df4['price']*100000/df4['total_sqft']
# print(df4)

# print(len(df4['location'].unique()))
#o/p :1304 & it is too much. it's called as Dimensionality curse. So we club together locations with less than 5 count to reduce the dimension.

df4['location'] = df4['location'].apply(lambda x: x.strip())
location_stat = df4['location'].value_counts()
# print(location_stat)

location_stat_less_than_10 = location_stat[location_stat<=10]
# print(location_stat_less_than_10)

# print(len(df4.location.unique()))     #o/p is 1293

df4['location']=df4['location'].apply(lambda x: 'other' if x in location_stat_less_than_10 else x)

# print(len(df4.location.unique()))    #o/p is 242. Hence reduced dimension

df4[df4['total_sqft']/df4['bhk']<300]  #Found out the outliners

df5 = df4[~(df4['total_sqft']/df4['bhk']<300)]    #Removed outliers
# print(df5)

# print(df5['price_per_sqft'].describe())       #Checking outliers

#To remove outliers in each location

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df6 = remove_pps_outliers(df5)
# print(df6.shape)

# To remove outliers where 2 bhk price > 3 bhk at same location

def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()

plot_scatter_chart(df6, "Rajaji Nagar")

#Fn to remove outliers where 2 bhk price > 3 bhk at same location

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df7 = remove_bhk_outliers(df6)
# df7 = df6.copy()
# print(df7.shape)

#To remove outlier using bathroom feature

# print(df7.bath.unique())
# print(df7[df7.bath>10])

df8 = df7[df7.bath<df7.bhk+2]
# print(df8.shape)

df9 = df8.drop(['size', 'price_per_sqft'],axis='columns')
# print(df9.head(5))

dummies = pd.get_dummies(df9.location)
# print(dummies)

df10 = pd.concat([df9, dummies.drop('other', axis = 'columns')], axis = 'columns')
# print(df10)

df11 = df10.drop('location', axis ='columns')
# print(df11)

X = df11.drop('price', axis = 'columns')
y = df11.price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X-train, y_train)
lr_clf.score(X_test, y_test)

