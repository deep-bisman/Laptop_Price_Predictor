import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
df=pd.read_csv('laptop_data.csv') 
#print(df.head(5))
#print(df.shape)
#print(df.info())
#print(df.describe())
#print(df.duplicated().sum())
#print(df.isnull().sum())
df1=df.copy()


# Here we do some EDA on data set
# 1) Unnamed column is not useful so we can drop it
df.drop(columns=['Unnamed: 0'],inplace=True)
#print(df.head(5))

# 2) ram and weight coulmn has both num and string value so we have to seperate and make them int
df['Ram']= df['Ram'].str.replace('GB','')
df['Weight']= df['Weight'].str.replace('kg','')
#print(df.head(5))

df['Ram']= df['Ram'].astype('int32')
df['Weight']= df['Weight'].astype('float32')
#print(df.info())



sns.displot(df['Price'],kde='True',color='red')
#plt.show()
# here we noticed that the price column has left skewed

df['Company'].value_counts().plot(kind='bar')
#plt.show()

sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
#plt.show()




df['TypeName'].value_counts().plot(kind='bar')
#plt.show()

sns.displot(df['TypeName'],kde='True',color='red')
plt.xticks(rotation='vertical')
#plt.show()

sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation='vertical')
#plt.show()



df['Inches'].value_counts().plot(kind='bar')
#plt.show()

sns.displot(df['Inches'],kde='True',color='red')
plt.xticks(rotation='vertical')
#plt.show()

sns.scatterplot(x=df['Inches'],y=df['Price'])
plt.xticks(rotation='vertical')
#plt.show()



#print(df['ScreenResolution'].value_counts())
df['Touchscreen']=df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
#print(df.head(20))
df['Touchscreen'].value_counts().plot(kind='bar')
#plt.show()

sns.barplot(x=df['Touchscreen'],y=df['Price'])
plt.xticks(rotation='vertical')
#plt.show()


df['IPS']=df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
#print(df.head(20))
df['IPS'].value_counts().plot(kind='bar')
#plt.show()
sns.barplot(x=df['IPS'],y=df['Price'])
plt.xticks(rotation='vertical')
#plt.show()

# new is a temporary dataframe

'''expand=True
👉 VERY IMPORTANT 🔥
False → gives list
True → gives separate columns'''

new=df['ScreenResolution'].str.split('x',n=1,expand=True)     
df['x_new']=new[0]
df['y_new']=new[1]
#print(df.head(10))

'''🔹 r'(\d+)'
👉 This is regex:
\d → digit (0–9)
+ → one or more digits
() → capture group (important)'''

df['x_new'] = df['x_new'].str.extract(r'(\d+)')
#print(df.head(10))

df['x_new']= df['x_new'].astype('int32')
df['y_new']= df['y_new'].astype('int32')
#print(df.info())
#print(df.corr(numeric_only=True)['Price'])

# here we noiced that inches.x_new and y_new has good co-relation with price and i decided to make a new column PPI
df['PPI']= (((df['x_new']**2) + (df['y_new']**2))**0.5/ df['Inches']).astype('float')
#print(df.head(10))
#print(df.info())
#print(df.corr(numeric_only=True)['Price'])

# now we extract imp info from Screenresol column now we can drop it

df.drop(columns=['ScreenResolution'],inplace=True)
df.drop(columns=['Inches','x_new','y_new'],inplace=True)
#print(df.head(10))
#print(df.shape)

df['CPU Name']=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
#print(df.head())


def fetch_processor(text):
    if text=='Intel Core i5' or text=='Intel Core i7' or text=='Intel Core i3':
        return text
    else:
        if text.split()[0]=='Intel':
            return 'Other Intel Processor'
        else:
            return 'Amd Processor'

df['CPU Brand']= df['CPU Name'].apply(fetch_processor)
#print(df.head(30))

df['CPU Brand'].value_counts().plot(kind='bar')
plt.xticks(rotation='vertical')
#plt.show()

sns.barplot(x=df['CPU Brand'],y=df['Price'])
plt.xticks(rotation='vertical')
#plt.show()

df.drop(columns=['Cpu','CPU Name'],inplace=True)
#print(df.head(5))



# now we have to evaluate memory column
#print(df['Memory'].value_counts())

# here we noticed that we have diffrent cateogries of ssd,hdd,hybrid and flash so,



# -----------------------------
# Step 1: Clean Memory column
# -----------------------------
df['Memory'] = df['Memory'].astype(str)
df['Memory'] = df['Memory'].str.replace(r'\.0', '', regex=True)
df['Memory'] = df['Memory'].str.replace('GB', '')
df['Memory'] = df['Memory'].str.replace('TB', '000')

# -----------------------------
# Step 2: Split into two parts
# -----------------------------
new = df['Memory'].str.split('+', n=1, expand=True)

df['first'] = new[0].str.strip()
df['second'] = new[1]

# -----------------------------
# Step 3: Handle missing values
# -----------------------------
df['second'] = df['second'].fillna('0')
df['second'] = df['second'].astype(str)

# -----------------------------
# Step 4: Identify storage types (Layer 1)
# -----------------------------
df['Layer1HDD'] = df['first'].apply(lambda x: 1 if 'HDD' in x else 0)
df['Layer1SSD'] = df['first'].apply(lambda x: 1 if 'SSD' in x else 0)
df['Layer1Hybrid'] = df['first'].apply(lambda x: 1 if 'Hybrid' in x else 0)
df['Layer1Flash_Storage'] = df['first'].apply(lambda x: 1 if 'Flash Storage' in x else 0)

# -----------------------------
# Step 5: Identify storage types (Layer 2)
# -----------------------------
df['Layer2HDD'] = df['second'].apply(lambda x: 1 if 'HDD' in x else 0)
df['Layer2SSD'] = df['second'].apply(lambda x: 1 if 'SSD' in x else 0)
df['Layer2Hybrid'] = df['second'].apply(lambda x: 1 if 'Hybrid' in x else 0)
df['Layer2Flash_Storage'] = df['second'].apply(lambda x: 1 if 'Flash Storage' in x else 0)

# -----------------------------
# Step 6: Extract only numbers
# -----------------------------
df['first'] = df['first'].str.replace(r'\D', '', regex=True)
df['second'] = df['second'].str.replace(r'\D', '', regex=True)

# -----------------------------
# Step 7: Convert to integer
# -----------------------------
df['first'] = df['first'].astype(int)
df['second'] = df['second'].astype(int)

# -----------------------------
# Step 8: Final storage features
# -----------------------------
df['HDD'] = (df['first'] * df['Layer1HDD'] + df['second'] * df['Layer2HDD'])
df['SSD'] = (df['first'] * df['Layer1SSD'] + df['second'] * df['Layer2SSD'])
df['Hybrid'] = (df['first'] * df['Layer1Hybrid'] + df['second'] * df['Layer2Hybrid'])
df['Flash_Storage'] = (df['first'] * df['Layer1Flash_Storage'] + df['second'] * df['Layer2Flash_Storage'])

# -----------------------------
# Step 9: Drop unnecessary columns
# -----------------------------
df.drop(columns=[
    'first', 'second',
    'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage',
    'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid', 'Layer2Flash_Storage'
], inplace=True)

# -----------------------------
# Done ✅
# -----------------------------
#print(df.head())



df.drop(columns=['Memory'],inplace=True)
#print(df.corr(numeric_only=True)['Price'])

# here we noticed that correlation between hybrid , flash storage with price is almost 0 so we have decided to drop it
df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)
#print(df.head())



#print(df['Gpu'].value_counts())
df['GPU Brand']=df['Gpu'].apply(lambda x: x.split()[0])
#print(df['GPU Brand'].value_counts())

# here we noticed that gpu brand column has one row having ARM GPU Brand so we have to discard it

df=df[df['GPU Brand']!='ARM']
#print(df['GPU Brand'].value_counts())

df.drop(columns=['Gpu'],inplace=True)
#print(df.shape)



#print(df['OpSys'].value_counts())
# here in opsys there are many cateogries so we have to group them

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'
    

df['os'] = df['OpSys'].apply(cat_os)  
df.drop(columns=['OpSys'],inplace=True)
#print(df.head())  
#print(df.corr(numeric_only=True)['Price'])


# here as we see price column is left skewed so we log it while training the model and result will convert back to exp.
sns.displot(np.log(df['Price']))
#plt.show()

df.to_csv('cleaned_laptop.csv',index=False)

#print(df.shape)

















