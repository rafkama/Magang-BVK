#!/usr/bin/env python
# coding: utf-8

# ## **Recommendation System**
# 
# Laporan kali ini berisi tentang eksperimen membuat sebuah sistem rekomendasi untuk Anime berdasarkan beberapa fitur yang dipilih. Sistem rekomendasi ini menggunakan Cosine Similarity untuk memberikan rekomendasi dan berfokus pada Content Based Filtering

# ## **Import Data**

# In[105]:


import pandas as pd
import numpy as np
df2=pd.read_csv('anime-dataset-2023.csv')


# In[106]:


# Drop rows where the 'Rating' column has the value 'Hentai'
df2 = df2[df2['Rating'] != 'Rx - Hentai']

# Reset index if you want the dataframe to have consecutive index values after dropping the rows
df2.reset_index(drop=True, inplace=True)


# In[107]:


df2


# In[108]:


df2.info()


# In[109]:


df2.isnull().sum().sort_values(ascending=False)


# ## **Weighted Rating**

# #### Weighted Rating dibutuhkan karena kemungkinan rating yang tinggi namun memiliki voter yang sedikit sehingga rating yang diberikan belum cukup untuk menggambarkan kualitas dari Anime tersebut

# In[110]:


# Convert 'Score' column to numeric, forcing invalid strings to NaN
df2['Score'] = pd.to_numeric(df2['Score'], errors='coerce')

# Calculate the mean, ignoring NaN values
C = df2['Score'].mean()

# Output the result
C


# In[111]:


df2['Scored By'] = pd.to_numeric(df2['Scored By'], errors='coerce')
m= df2['Scored By'].quantile(0.9)
m


# In[112]:


q_movies = df2.copy().loc[df2['Scored By'] >= m]
q_movies.shape


# In[113]:


def weighted_rating(x, m=m, C=C):
    v = x['Scored By']
    R = x['Score']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[114]:


q_movies['Weighted Rating'] = q_movies.apply(weighted_rating, axis=1)


# In[115]:


#Sort movies based on score calculated above
q_movies = q_movies.sort_values('Weighted Rating', ascending=False)

#Print the top 15 movies
q_movies[['Name', 'Scored By', 'Score', 'Weighted Rating']].head(10)


# In[116]:


df2['Weighted Rating'] = df2.apply(weighted_rating, axis=1)


# In[117]:


df2


# In[118]:


import matplotlib.pyplot as plt

# Remove rows where 'Popularity' is 0 or null
df2_filtered = df2[(df2['Popularity'] > 0) & (df2['Popularity'].notnull())]

# Sort the remaining data by 'Popularity'
pop = df2_filtered.sort_values('Popularity', ascending=True)

# Create the plot
plt.figure(figsize=(12,4))
plt.barh(pop['Name'].head(6), pop['Popularity'].head(6), align='center', color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Anime")
plt.show()


# ## **Conteny Based Filtering**

# In[119]:


df2['Synopsis'].head(5)


# In[120]:


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df2['Synopsis'] = df2['Synopsis'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['Synopsis'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[121]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[122]:


#Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['Name']).drop_duplicates()


# In[181]:


from IPython.display import display, Image

def get_recommendations(Name, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[Name]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Filter recommendations where Weighted Rating > 6.9
    filtered_recommendations = df2.iloc[movie_indices]
    filtered_recommendations = filtered_recommendations[filtered_recommendations['Weighted Rating'] > 6.5]
    filtered_recommendations = filtered_recommendations[1:15]

    # Display the names and images of the filtered recommendations
    for index, row in filtered_recommendations.iterrows():
        print(f"Title: {row['Name']}")
        display(Image(url=row['Image URL']))

    return filtered_recommendations[['Name', 'Image URL']]


# In[182]:


get_recommendations('Clannad')


# ## **Credits, Genres and Keywords Based Recommender**

# In[126]:


pip install nltk


# In[127]:


import nltk
nltk.download('omw-1.4')


# In[128]:


import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (run these once if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to process text (tokenize, remove stopwords, punctuation, and lemmatize)
def process_synopsis(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    keywords = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Join keywords with a comma separator
    return ','.join(keywords)

# Apply the function to the 'Synopsis' column and create a new column 'keywords'
df2['keywords'] = df2['Synopsis'].apply(process_synopsis)

# View the first few rows to verify
df2[['Synopsis', 'keywords']].head()


# In[129]:


df2[['Studios', 'Producers', 'keywords', 'Genres']].head()


# In[130]:


# Define a function to split comma-separated strings into lists
def split_to_list(val):
    if isinstance(val, str):
        return val.split(',')  # Split string by comma
    return val  # If not a string, return the original value

# Apply the function to each column
features = ['Studios', 'Producers', 'keywords', 'Genres']
for feature in features:
    df2[feature] = df2[feature].apply(split_to_list)

# Check the first few rows to verify the result
df2[features].head()


# In[131]:


# No need to use literal_eval here
features = ['Studios', 'Producers', 'keywords', 'Genres']

# Verify that each feature has been converted to a list
for feature in features:
    df2[feature] = df2[feature].apply(split_to_list)

# Check the first few rows to verify
df2[features].head()


# In[132]:


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# In[133]:


# Apply clean_data function to your features.
features = ['Studios', 'Producers', 'keywords', 'Genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)


# In[134]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['Producers']) + ' ' + ' '.join(x['Studios'] if isinstance(x['Studios'], list) else [x['Studios']]) + ' ' + ' '.join(x['Genres'])

# Apply the function to create the 'soup' column
df2['soup'] = df2.apply(create_soup, axis=1)

# Check the first few rows
df2[['keywords', 'Producers', 'Studios', 'Genres', 'soup']].head()


# In[135]:


# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])


# In[136]:


# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[137]:


# Reset index of our main DataFrame and construct reverse mapping as before
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['Name'])


# In[192]:


get_recommendations('Perfect Blue', cosine_sim2)


# In[193]:


get_recommendations('Perfect Blue')


# In[ ]:




