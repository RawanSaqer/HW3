import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

pd.__version__

pd.show_versions()



df = pd.read_csv('Iris.csv')
df

plt.figure(figsize=(8, 8))

df['P_Width'].plot.hist()

plt.xlabel('Height')
plt.ylabel('Count')

plt.show()


plt.figure(figsize=(8, 6))

sns.kdeplot(df['P_Width'], shade=True, color='g')

plt.show()

df = pd.DataFrame({"S_Width"  :df.S_Length,
                   "S_Length"    :df.S_Width,
                   "P_Length"  :df.P_Length,
                   "P_Width" :df.P_Width})

df.plot.scatter("S_Length", "S_Width", s = df.P_Length * 30, c = df.P_Width)

df = pd.DataFrame({"S_Length"    :df.S_Width,
                   "P_Length"  :df.P_Length,
                   "P_Width" :df.P_Width})
                    

ax = df.plot.bar("P_Width","S_Length", color = "green")
df.plot.line("P_Width", "P_Length", secondary_y = True, ax = ax)
ax.set_xlim((-1,12))

df.describe()

df.info()

df.iloc[:3]

df.head(3)

df.loc[:,['S_Length','P_Length']]

df.loc[df.index[[3, 4, 8]], ['S_Length','P_Length']]

df[df['S_Length'] > 5]

df[df['P_Length'].isnull()]

df[(df['Type_2'] == 'Iris-setosa') & (df['S_Length'] <5)]

df[df['S_Length'].between(4, 5)]

df['S_Length'].sum()

df.groupby('P_Length')['P_Width'].mean()

df.loc['M'] = [151,5.1 , 3.1 ,5.1,6,'Iris-virginica'] 
df

df = df.drop('M')
df

df['Type'].value_counts()

'''
#????????????????????????????
df.sort_values(by=['S_Length'], inplace=True)
df
'''

df.sort_values(by=['S_Length', 'S_Width'], ascending=[False, True])
df

df['Type'] = df['Type'].map({'Iris-setosa': True, 'Iris-versicolor': False})
df.head(100)

df.loc['Rawan'] = [151,11 , 3.1 ,5.1,6,'Iris-virginica'] 
df

# Check for NaN in Pandas DataFrame

df['S_Length'] = df['S_Length'].replace(11, 9)
df

# DataFrames

df.isnull().values.any()

df.isnull().sum().sum()

df['S_Length'].isnull().values.any()

df['S_Length'].isnull().sum()

df['S_Width'].isnull().sum()

df['P_Length'].isnull().sum()

df['P_Width'].isnull().sum()

df['Type_2'].isnull().sum()

df.isnull().sum()

df.isnull().sample(150)

# Cleaning Data

df['P_Length'] = df['P_Length'].interpolate().astype(int)
df

df['P_Length'].isnull().sum()

df['P_Length'].sample(100)

# Plotting
- Visualize trends and patterns in data

df = pd.DataFrame({"xs":df.S_Length, "ys":df.S_Width})
df.plot.scatter("xs", "ys", color = "purple", marker = "*")

data = {'name': ['Rawan','Maram', 'Reem','Jory','Tala'],
        'age': [25, 27, 29, 17, 25]}

labels = ['a', 'b', 'c', 'd', 'e']

z= pd.DataFrame(data, index=labels)
z

z.loc['b', 'age'] = 26
z

z.loc[z['age'].shift() != z['age']]
z.drop_duplicates(subset='age')


# more information

SA = pd.DataFrame(np.random.random(size=(5, 3)))

SA.sub(SA.mean(axis=1), axis=0)

rr= pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))

rr.sum().idxmin()

ww = pd.DataFrame(np.random.randint(0, 2, size=(10, 3)))

len(ww) - ww.duplicated(keep=False).sum()

# or perhaps more simply...

len(ww.drop_duplicates(keep=False))

nan = np.nan

data = [[0.04,  nan,  nan, 0.25,  nan, 0.43, 0.71, 0.51,  nan,  nan],
        [ nan,  nan,  nan, 0.04, 0.76,  nan,  nan, 0.67, 0.76, 0.16],
        [ nan,  nan, 0.5 ,  nan, 0.31, 0.4 ,  nan,  nan, 0.24, 0.01],
        [0.49,  nan,  nan, 0.62, 0.73, 0.26, 0.85,  nan,  nan,  nan],
        [ nan,  nan, 0.41,  nan, 0.05,  nan, 0.61,  nan, 0.48, 0.68]]

columns = list('abcdefghij')

ra1 = pd.DataFrame(data, columns=columns)


(ra1.isnull().cumsum(axis=1) == 3).idxmax(axis=1)

ra3 = pd.DataFrame({'grps': list('aaabbcaabcccbbc'), 
                   'vals': [12,345,3,1,45,14,4,52,54,23,235,21,57,3,87]})

ra3.groupby('grps')['vals'].nlargest(3).sum(level=0)

ra4 = pd.DataFrame(np.random.RandomState(8765).randint(1, 101, size=(100, 2)), columns = ["A", "B"])

ra4.groupby(pd.cut(ra4['A'], np.arange(0, 101, 10)))['B'].sum()

ra5 = pd.DataFrame({'X': [7, 2, 0, 3, 4, 2, 5, 0, 3, 4]})

izero = np.r_[-1, (ra5 == 0).values.nonzero()[0]]  # indices of zeros
idx = np.arange(len(ra5))
y = ra5['X'] != 0
ra5['Y'] = idx - izero[np.searchsorted(izero - 1, idx) - 1]
ra5

ra6 = pd.DataFrame(np.random.RandomState(30).randint(1, 101, size=(8, 8)))

ra6.unstack().sort_values()[-3:].index.tolist()

letters = ['A', 'B', 'C']
numbers = list(range(10))

mi = pd.MultiIndex.from_product([letters, numbers])
s = pd.Series(np.random.rand(30), index=mi)
s

s.index.is_lexsorted()
