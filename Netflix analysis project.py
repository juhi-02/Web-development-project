import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Netflix = pd.read_csv("D:/web dev/web dev assignment/netflix_titles.csv")

print("Printing first 10 rows to become familiar with the dataset.\n".upper())
print(Netflix.head(10))

print("\nShape of the dataset.\n".upper())
print(Netflix.shape)

print("\nDataset info\n".upper())
Netflix.info()

print("\nDataset description\n".upper())
print(Netflix.describe(include='all'))


print("Duplicate show_id count before:", Netflix['show_id'].duplicated().sum())

#Drop duplicate show_id rows
Netflix = Netflix.drop_duplicates(subset='show_id')

print("Duplicate show_id count after:", Netflix['show_id'].duplicated().sum())

#Drop the description column
Netflix = Netflix.drop('description',axis=1)

print("Printing first 10 rows again".upper())
print(Netflix.head(10))

print("Missing values in each column:\n")
print(Netflix.isnull().sum())

Netflix['country'] = Netflix['country'].fillna("Unknown")

Netflix['director'] = Netflix['director'].fillna("No director listed")

Netflix['duration_minutes'] = Netflix.apply(
    lambda x: int(x['duration'].split()[0])
    if x['type'] == 'Movie' and pd.notnull(x['duration'])
    else np.nan,
    axis=1
)


Netflix['seasons'] = Netflix.apply(
    lambda x: int(x['duration'].split()[0])
    if x['type'] == 'TV Show' and pd.notnull(x['duration'])
    else np.nan,
    axis=1
)


print("\nNew engineered columns preview:\n")
print(Netflix[['type', 'duration', 'duration_minutes', 'seasons']].head(10))

Netflix['Is_Recent'] = Netflix['release_year'].apply(
    lambda x: 1 if x >= 2015 else 0
)

print("Binary feature created".upper())
print(Netflix[['title', 'release_year', 'Is_Recent']].head(10))   #verifying above code



print("PLOTTING DATA\n")

print("UNIVARIATE ANALYSIS")
sns.countplot(x='type', data=Netflix)       #countplot() draws a bar for each type(Movies and tv show)
plt.title("Count of Movies vs TV Shows on Netflix")
plt.xlabel("Type")   #This appears on the x-axis of count plot
plt.ylabel("Count")  #This appears on the y-axis of count plot
plt.show()


plt.hist(Netflix['release_year'], bins=10, color = 'pink')  #this plots a histogram for the data in the column 'release year'
plt.title("Distribution of Release Years")
plt.xlabel("Release Year")      #This appears on the x-axis of count plot
plt.ylabel("Number of Titles")  #This appears on the y-axis of count plot
plt.show()


print("BIVARIATE ANALYSIS")
#This is bivariate analysis because it involves analysing two variables

top_countries = Netflix['country'].value_counts().head(10)   #Counts how many times each country appears and sorts the result in descending order. It keeps only 10 countries as specified.

sns.barplot(x=top_countries.values, y=top_countries.index, color = 'yellow')   #seaborn library is used to visualize a barplot. Number of titles on x-axis and country names(index) on y-axis.
plt.title("Number of releases per country (Top 10)")
plt.xlabel("Number of Titles")      #This appears on the x-axis of count plot
plt.ylabel("Country")               #This appears on the y-axis of count plot
plt.show()


movies = Netflix[Netflix['type'] == 'Movie']   #Keeps only movie rows in 'movies' variable from column 'type'

sns.boxplot(x='Is_Recent', y='duration_minutes', data=movies, color = 'red')   #Boxplot is plotted which compares movie durations between recent and older movies.
plt.title(" Comparing duration_minutes between movies from recent vs older years ")
plt.xlabel("Is_Recent (1 = Recent, 0 = Older)")       #This appears on the x-axis of count plot
plt.ylabel("Duration (minutes)")                      #This appears on the y-axis of count plot
plt.show()


print("CORRELATION HEATMAP:")
numeric_data = Netflix[['release_year', 'duration_minutes', 'seasons', 'Is_Recent']]   #numeric_data contains only numbers. This is important because Correlation works only on numerical data
plt.figure(figsize=(6,4))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')  #.corr() - calculates relationships, heatmap - visualizes them, annot=True - shows numbers, cmap - sets color scheme
plt.title("Correlation Heatmap of Numerical Features")
plt.show()












