#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import missingno as msno
from datetime import datetime
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Persist all generated artifacts to results/ ---
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

_fig_counter = {"n": 0}
_orig_plt_show = plt.show


def _save_and_show(*args, **kwargs):
    _fig_counter["n"] += 1
    fig_path = os.path.join(FIGURES_DIR, f"fig_{_fig_counter['n']:03d}.png")
    try:
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    except Exception as exc:
        print(f"[warn] could not save {fig_path}: {exc}")
    return _orig_plt_show(*args, **kwargs)


# Monkey-patch plt.show so every figure in the pipeline is auto-saved.
plt.show = _save_and_show


def save_table(df, name):
    """Write a DataFrame/Series result to results/tables/<name>.csv."""
    path = os.path.join(TABLES_DIR, f"{name}.csv")
    try:
        if isinstance(df, pd.Series):
            df.to_frame().to_csv(path)
        else:
            df.to_csv(path, index=False)
        print(f"[saved] {path}")
    except Exception as exc:
        print(f"[warn] could not save {path}: {exc}")
# --- end results/ wiring ---


# Loading the data set 

# In[2]:


# Place reddit_opinion_PSE_ISR.csv in a `data/` folder next to this script,
# or set the REDDIT_DATA_PATH environment variable to its full path.
DEFAULT_DATA_PATH = os.path.join("data", "reddit_opinion_PSE_ISR.csv")
data_path = os.environ.get("REDDIT_DATA_PATH", DEFAULT_DATA_PATH)

# Load the dataset
df_orginal = pd.read_csv(data_path)

df_orginal.head(5)


# Getting Closer Look

# In[3]:


df_orginal.shape


#  Dataset has 24 features with more than 1820000 records
# 

# In[4]:


df_orginal.sample(5)


# In[5]:


df_orginal.columns


# In[6]:


df_orginal.describe()


# In[7]:


df = df_orginal


# In[8]:


df.dtypes


# Convert Date Fields from Object to Datetime

# In[9]:


# Convert date-related columns to datetime format
df['created_time'] = pd.to_datetime(df['created_time'])
df['user_account_created_time'] = pd.to_datetime(df['user_account_created_time'])
df['post_created_time'] = pd.to_datetime(df['post_created_time'])

# Confirm that the conversions have been applied
print("\nData types after conversion:")
print(df.dtypes)


# In[10]:


df.tail(5)


# Let's review the dates range in the dataset

# In[11]:


from IPython.display import display, HTML
display(HTML(f"<h5><b style='color:red'>The Oldest Post Date: </b>{df.post_created_time.min()}</h5>"))
display(HTML(f"<h5><b style='color:red'>Post Title: </b>{df.loc[df['post_created_time'] == df.post_created_time.min()].values[0][19] }</h5>"))
display(HTML(f"<h5><b style='color:red'>Subreddit: </b>{df.loc[df['post_created_time'] == df.post_created_time.min()].values[0][3] }<hr></h5>"))

display(HTML(f"<h5><b style='color:red'>The Newest Post Date: </b>{df.post_created_time.max()}</h5>"))
display(HTML(f"<h5><b style='color:red'>Post Title: </b>{df.loc[df['post_created_time'] == df.post_created_time.max()].values[0][19] }</h5>"))
display(HTML(f"<h5><b style='color:red'>Subreddit: </b>{df.loc[df['post_created_time'] == df.post_created_time.max()].values[0][3] }</h5>"))


# In[12]:


display(HTML(f"<h5><b style='color:red'>The Oldest Comment Date: </b>{df.post_created_time.min()}</h5>"))
display(HTML(f"<h5><b style='color:red'>Comment: </b>{df.loc[df['created_time'] == df.created_time.min()].values[0][2] }</h5>"))
display(HTML(f"<h5><b style='color:red'>Subreddit: </b>{df.loc[df['post_created_time'] == df.post_created_time.min()].values[0][3] }<hr></h5>"))

display(HTML(f"<h5><b style='color:red'>The Newest Comment Date: </b>{df.post_created_time.max()}</h5>"))
display(HTML(f"<h5><b style='color:red'>Comment: </b>{df.loc[df['created_time'] == df.created_time.max()].values[0][2] }</h5>"))
display(HTML(f"<h5><b style='color:red'>Subreddit: </b>{df.loc[df['created_time'] == df.created_time.max()].values[0][3] }</h5>"))


# Drop Records Before 7th Oct 2023
# 

# On October 7, 2023, Hamas initiated what it called "Operation Al-Aqsa Flood" followed shortly by the IDF launching a military operation in Gaza. Given the significance of these events, our data analysis will focus primarily on information starting from October 7th.

# In[13]:


print('Len. of data before 2023-10-07:' ,len(df))
start_date = pd.to_datetime('2023-10-07')

# Select data(posts+comments) starting from '2023-10-07'
filtered_df = df[(df['post_created_time'] >= start_date) & (df['created_time'] >= start_date)].copy()
print('Len. of data After 2023-10-07:',len(filtered_df))
print('Num. of dropped rows:',len(df)-len(filtered_df))


# Data Per Processing 

# Check Null Values

# In[14]:


# Check for null values in the dataset
null_counts = filtered_df.isnull().sum()
print("Null Values in Each Column:\n", null_counts)

# Calculate the percentage of missing values
null_percentage = (null_counts / len(filtered_df)) * 100
print("Percentage of Null Values in Each Column:\n", null_percentage)


# Let's visualize the missing values spread in the dataset

# In[15]:


#  Missingness Matrix
msno.bar(filtered_df, figsize=(12, 4),color=(0.3, 0.3, 0.5))


# In[16]:


# Fill missing self_text with a placeholder
filtered_df['self_text'] = filtered_df['self_text'].fillna('')


# In[17]:


filtered_df[['user_comment_karma', 'user_total_karma', 'user_awardee_karma', 'user_awarder_karma', 'user_link_karma']] = filtered_df[['user_comment_karma', 'user_total_karma', 'user_awardee_karma', 'user_awarder_karma', 'user_link_karma']].fillna(0)


# In[18]:


# Calculate the median timestamp
median_timestamp = filtered_df['user_account_created_time'].median()

# Fill missing values with the median timestamp
filtered_df['user_account_created_time'] = filtered_df['user_account_created_time'].fillna(median_timestamp)


# In[19]:


# Fill missing 'post_self_text' with an empty string
filtered_df['post_self_text'] = filtered_df['post_self_text'].fillna('')


# Checking again for null values after handeling the carefully

# In[20]:


# Check for null values in the dataset
null_counts = filtered_df.isnull().sum()
print("Null Values in Each Column:\n", null_counts)


# Check Duplicate Values

# In[21]:


# Check for duplicates based on 'post_title' and 'post_self_text'
duplicates = filtered_df[filtered_df.duplicated(subset=['post_title', 'post_self_text'], keep=False)]

# Display the number of duplicate entries and some examples
print(f"Number of duplicate rows based on 'post_title' and 'post_self_text': {len(duplicates)}")
duplicates.head() 


# In[22]:


# Check for duplicates based on 'post_title', 'post_self_text', 'author_name', and 'created_time'
duplicates_extended = filtered_df[filtered_df.duplicated(subset=['post_title', 'post_self_text', 'author_name', 'created_time'], keep=False)]

# Display the number of duplicate entries and some examples
print(f"Number of extended duplicate rows based on 'post_title', 'post_self_text', 'author_name', and 'created_time': {len(duplicates_extended)}")
duplicates_extended.head() 


# In[23]:


# Keep the entry with the highest score for each set of duplicates
filtered_df_no_duplicates = filtered_df.loc[
    filtered_df.groupby(['post_title', 'post_self_text', 'author_name', 'created_time'])['score'].idxmax()
]


# In[24]:


# Remove duplicates, keeping only the first occurrence
filtered_df_no_duplicates = filtered_df.drop_duplicates(subset=['post_title', 'post_self_text', 'author_name', 'created_time'], keep='first')


# In[25]:


# Merge duplicates by summing up engagement metrics
filtered_df_merged = filtered_df.groupby(['post_title', 'post_self_text', 'author_name', 'created_time'], as_index=False).agg({
    'ups': 'sum',
    'downs': 'sum',
    'score': 'mean',  
    'comment_id': 'first',  
    'subreddit': 'first',  
})


# In[26]:


# Verify the number of rows after handling duplicates
print(f"Number of rows after handling duplicates: {len(filtered_df_no_duplicates)}")

# Check for remaining duplicates to ensure none are left
remaining_duplicates = filtered_df_no_duplicates.duplicated(subset=['post_title', 'post_self_text', 'author_name', 'created_time']).sum()
print(f"Number of remaining duplicates: {remaining_duplicates}")


# In[27]:


# Check the dublicated comments in dataset
print(f'Duplicate Comments: {filtered_df_no_duplicates[["self_text"]].duplicated().sum()}')


# In[28]:


# Filter the dataset to get only the duplicated rows based on 'self_text'
df_duplicate_comments = filtered_df_no_duplicates[filtered_df_no_duplicates.duplicated(subset=['self_text'], keep=False)]

# Count the occurrences of duplicated 'self_text'
duplicate_comment_counts = df_duplicate_comments['self_text'].value_counts().reset_index()

duplicate_comment_counts.columns = ['self_text', 'count']

# Show the top 15 most duplicated 'self_text' entries
print("Top 15 most duplicated comments:")
print(duplicate_comment_counts.head(20))


# In[29]:


# Define a list of short generic and moderation comments to remove
generic_and_moderation_comments = [
    "Yes", "Source?", "Your account was detected as a ban evading account...", 
    "Lol", "No", "Exactly", "Thank you", "Yes.", 
    "This has been removed for breaking the Reddit rules...", 
    "Your content has been removed for violating...", "What?", 
    "Based", "lol", "No.", "Why?","Thanks"
]

# Remove these comments from the dataset
filtered_df_no_generic_comments = filtered_df_no_duplicates[~filtered_df_no_duplicates['self_text'].isin(generic_and_moderation_comments)]

# Check the number of rows remaining after removing generic and moderation comments
print(f"Number of rows after removing generic and moderation comments: {len(filtered_df_no_generic_comments)}")


# In[30]:


df=filtered_df_no_generic_comments


# Shape after handeling the duplicate values 

# In[31]:


# Check the shape of the dataset after removing these comments
print(f"Shape of dataset after removing moderation/policy comments: {df.shape}")


# Correlation Analysis

# In[32]:


# Selecting only numerical features for correlation matrix
numerical_cols = [
    'score', 'ups', 'downs', 'user_awardee_karma', 'user_awarder_karma',
    'user_link_karma', 'user_comment_karma', 'user_total_karma',
    'post_score', 'post_upvote_ratio', 'post_thumbs_ups', 'post_total_awards_received','controversiality'
]

correlation_matrix = df[numerical_cols].corr()

# Plotting the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Analyze Correlation Values
print("Correlation Matrix:\n", correlation_matrix)


# Let's drop features that don't provide meaningful insights

# In[33]:


#Columns to be removed due to low correlation and lack of relevance
columns_to_drop = [
    'ups',                   # Perfectly correlated with score
    'user_awardee_karma',    # Low correlation and redundant
    'user_awarder_karma',    # Low correlation and redundant
    'user_link_karma',       # High correlation with user_total_karma
    'user_comment_karma',    # High correlation with user_total_karma
    'downs',                 # Mostly NaN or zero
    'post_thumbs_ups',       # Similar to post_score
    'post_total_awards_received'  # NaN correlation, likely sparsely populated
]
# Check if columns exist in the dataframe and drop them
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# Display the updated columns of the dataframe
print("Remaining columns after dropping irrelevant features:\n", df.columns)


# In[34]:


df.describe().T


# In[35]:


df.describe(include='object').T


# In[36]:


# Remove rows with deleted authors
df = df[df['author_name'] != '[deleted]']

# Verify the result
df.describe(include='object').T


# In[37]:


df.shape


# In[38]:


df.head()


# Exploratory Data Analysis 

# In[39]:


print("Number of unique authors:",  df['author_name'].nunique())


# In[40]:


# Calculate the difference between max_date and min_date
date_difference = df.post_created_time.max() - df.post_created_time.min()

# Calculate years, months, and days
years = date_difference.days // 365
months = (date_difference.days % 365) // 30
days = (date_difference.days % 365) % 30

# Generate result string
result = f"Data covers a period of {years} years, {months} months, {days} days," \
         f" from ({df.post_created_time.dt.date.min()}) till ({df.post_created_time.dt.date.max()})"

print(result)


# In[41]:


# Identify the post with the highest score in the dataset
highest_score_post = pd.DataFrame(df.loc[df['post_score'].idxmax()][['post_title', 'post_score', 'subreddit', 'post_created_time']]).T

highest_score_post


# In[42]:


# Identify the comment with the highest score in the dataset
highest_score_comment = pd.DataFrame(df.loc[df['score'].idxmax()][['self_text', 'post_title', 'score', 'subreddit', 'created_time']]).T

# Display the result
highest_score_comment


# In[43]:


# Calculate the most popular subreddits
popular_subreddits = df['subreddit'].value_counts().reset_index()
popular_subreddits.columns = ['subreddit', 'count']
save_table(popular_subreddits, "popular_subreddits")

# Display the result
popular_subreddits


# In[44]:


# Plot the most popular subreddits
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x='count', y='subreddit', data=popular_subreddits, palette='magma')

# Annotate each bar with its count
for p in barplot.patches:
    barplot.annotate(format(p.get_width(), '.0f'),
                     (p.get_width(), p.get_y() + p.get_height() / 2.),
                     va='center', ha='left', xytext=(10, 0),
                     textcoords='offset points')

plt.title('Top Popular Subreddits')
plt.xlabel('Count')
plt.ylabel('Subreddit')
plt.show()


#  Let's explore the commenting Activity for the most popular Subreddits over time.
# 

# In[45]:


# Select top 5 popular Subreddits
most_popular_subreddits = popular_subreddits.head(5)['subreddit'].tolist()

# Filter DataFrame for popular subreddits
most_popular_subreddits_data = df[df['subreddit'].isin(most_popular_subreddits)]

# Group by 'subreddit' and 'created_time', count comments, and unstack to have each subreddit as a separate line
grouped_data = most_popular_subreddits_data.groupby(['subreddit', df['created_time'].dt.date]).size().unstack(level=0)

# Plotting each subreddit
plt.figure(figsize=(15, 6))
for subreddit in grouped_data.columns:
    sns.lineplot(x=grouped_data.index, y=grouped_data[subreddit], label=subreddit)

# Adjust x-axis ticks
plt.xticks(grouped_data.index[::6], grouped_data.index[::6], rotation=60, ha='right')
# Adding labels and title
plt.title('Commenting Activity by Subreddit over Dates')
plt.xlabel('Date')
plt.ylabel('Number of Comments')
plt.legend(title='Subreddit', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# In[46]:


# Filter DataFrame for the specified subreddits
df_subreddits = df[df['subreddit'].isin(most_popular_subreddits)]

# Group by 'subreddit' and 'created_time', calculate daily comment count
daily_comment_counts = df_subreddits.groupby(['subreddit', df['created_time'].dt.date]).size().reset_index(name='comment_count')

# Calculate daily commenting rate (average comments per day)
daily_comment_rates = daily_comment_counts.groupby('subreddit')['comment_count'].mean().reset_index(name='average_comment_rate')

# Set the style of the visualization
sns.set(style="whitegrid")

# Create a swarm plot
plt.figure(figsize=(12, 5))
sns.swarmplot(x='average_comment_rate', y='subreddit', data=daily_comment_rates, palette='cubehelix', size=15)

# Set title and labels
plt.title('Distribution of Average Comment Rates Across Subreddits', fontsize=16)
plt.xlabel('Average Comment Rate', fontsize=14)
plt.ylabel('Subreddit', fontsize=14)

plt.show()

daily_comment_rates


# Exploring the Most Engaging Subreddits | IsraelPalestine

# It's seen that the most engaging subreddit appears to be "IsraelPalestine" as evidenced by the substantial comment counts and varying durations of engagement.Let's take a swift look to gain insights!

# In[47]:


# Count unique active users in the 'IsraelPalestine' subreddit
unique_users_count = df[df['subreddit'] == 'IsraelPalestine']['author_name'].nunique()

display(HTML(f"<h5>Number of unique users in IsraelPalestine subreddit: <b style='color:red'>{unique_users_count}</b></h5>"))


# In[48]:


# Filter posts related to 'IsraelPalestine' subreddit 
posts_israel_palestine = df[(df['subreddit'] == 'IsraelPalestine') & (df['post_self_text'].notnull())]

# Drop duplicate posts
unique_posts = posts_israel_palestine.drop_duplicates(subset='post_self_text')
posts_count = len(unique_posts)

display(HTML(f"<h5>Number of unique posts in IsraelPalestine subreddit: <b style='color:red'>{posts_count}</b></h5>"))


# In[49]:


# Filter comments related to 'IsraelPalestine' subreddit
comments_israel_palestine = df[(df['subreddit'] == 'IsraelPalestine') & (df['self_text'].notnull())]

# Drop duplicate comments
unique_comments = comments_israel_palestine.drop_duplicates(subset='self_text')

# Get the counts
comments_count = len(unique_comments)

display(HTML(f"<h5>Number of unique comments in IsraelPalestine subreddit: <b style='color:red'>{comments_count}</b></h5>"))


#  Temporal Trends

# In[50]:


# Create a copy of the DataFrame
filtered_df_unique = df.copy()

# Convert 'created_time' to datetime format and extract the month
filtered_df_unique['created_time'] = pd.to_datetime(filtered_df_unique['created_time']).dt.to_period('M')

# Get the count of comments for each month
monthly_comment_counts = filtered_df_unique['created_time'].value_counts().sort_index()

# Create a DataFrame from the monthly counts
monthly_counts_df = pd.DataFrame({'Month': monthly_comment_counts.index.to_timestamp(), 'Comment_Creations': monthly_comment_counts.values})

# Plotting using Seaborn
plt.figure(figsize=(20, 6))
lineplot = sns.lineplot(x='Month', y='Comment_Creations', data=monthly_counts_df, palette='viridis')

# Add labels to the data points
for x, y in zip(monthly_counts_df['Month'], monthly_counts_df['Comment_Creations']):
    plt.text(x, y, str(y), ha='center', va='bottom', fontsize=10, rotation=45)

# Set the x-axis tick labels
date_ticks = monthly_counts_df['Month'][::1]  # Display every month
plt.xticks(date_ticks, rotation=45, ha='right')

# Highlight a specific period (e.g., November 2023)
highlight_start_date = pd.Timestamp('2023-11-01')
highlight_end_date = pd.Timestamp('2023-11-30')

lineplot.axvspan(highlight_start_date, highlight_end_date, color='lightcoral', alpha=0.3, label='Highlighted Period')

plt.title('Temporal Trends - Comments per Month | All Subreddits')
plt.xlabel('Month')
plt.ylabel('Total Number of Comments')
plt.legend(title='Legend', title_fontsize='12', loc='upper left')
plt.tight_layout()
plt.show()


# In[51]:


df_copy=df


# Lets Normailze the text 

# In[52]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# Install with: pip install contractions  (see requirements.txt)

import contractions
# Function to clean and normalize text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = contractions.fix(text)  # Expand contractions
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters and punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\x00-\x7f]', '', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text

# Function to remove stopwords
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

# Function to lemmatize text
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


# In[53]:


# Apply the complete cleaning function to 'self_text'
df['cleaned_self_text'] = df['self_text'].apply(clean_text)
 
# Remove stopwords from 'cleaned_self_text'
df['cleaned_self_text'] = df['cleaned_self_text'].apply(remove_stopwords)

# Lemmatize 'cleaned_self_text'
df['cleaned_self_text'] = df['cleaned_self_text'].apply(lemmatize_text)

# Check the cleaned text output for 'self_text'
print(df[['self_text', 'cleaned_self_text']].head())



# In[54]:


# Apply the same cleaning process to 'post_self_text' 
df['cleaned_post_self_text'] = df['post_self_text'].apply(clean_text)
df['cleaned_post_self_text'] = df['cleaned_post_self_text'].apply(remove_stopwords)
df['cleaned_post_self_text'] = df['cleaned_post_self_text'].apply(lemmatize_text)

# Apply the same cleaning process to 'post_title' 
df['cleaned_post_title'] = df['post_title'].apply(clean_text)
df['cleaned_post_title'] = df['cleaned_post_title'].apply(remove_stopwords)
df['cleaned_post_title'] = df['cleaned_post_title'].apply(lemmatize_text)

# Check the cleaned text output for 'post_self_text' and 'post_title'
df[['post_self_text', 'cleaned_post_self_text', 'post_title', 'cleaned_post_title']].head(5)


# In[55]:


df.head()


# Visualizations  for the text columns 

# In[56]:


from collections import Counter

# Sample a smaller subset of comments
df_sample = df.sample(n=100000, random_state=42)  # Adjust based on available memory

# Split and count the words as before
all_words = ' '.join(df_sample['cleaned_self_text']).split()
word_counts = Counter(all_words)

# Convert the counter to a DataFrame for plotting
word_counts_df = pd.DataFrame(word_counts.most_common(10), columns=['word', 'count'])

# Plot the top 10 most frequent words
plt.figure(figsize=(10, 6))
sns.barplot(x='count', y='word', data=word_counts_df, palette='viridis')
plt.title('Top 10 Most Frequent Words')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.show()


# In[57]:


from wordcloud import WordCloud

# Create a word cloud using the most frequent words
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(word_counts)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Most Frequent Words')
plt.show()


# In[58]:


# Count the number of comments per subreddit
subreddit_counts = df['subreddit'].value_counts()

# Plot the top 10 most active subreddits
plt.figure(figsize=(10, 6))
subreddit_counts.head(10).plot(kind='bar')
plt.title('Top 10 Most Active Subreddits')
plt.xlabel('Subreddit')
plt.ylabel('Number of Comments')
plt.show()


# In[59]:


# Calculate the average score per subreddit
subreddit_avg_score = df.groupby('subreddit')['score'].mean().sort_values(ascending=False)

# Plot the top 10 subreddits by average score
plt.figure(figsize=(10, 6))
subreddit_avg_score.head(10).plot(kind='bar')
plt.title('Top 10 Subreddits by Average Comment Score')
plt.xlabel('Subreddit')
plt.ylabel('Average Score')
plt.show()


# In[60]:


# Identify top 10 users by the number of comments
top_commenters = df['author_name'].value_counts().head(10)

# Plot the top 10 users by number of comments
plt.figure(figsize=(10, 6))
top_commenters.plot(kind='bar')
plt.title('Top 10 Users by Number of Comments')
plt.xlabel('Username')
plt.ylabel('Number of Comments')
plt.show()


# In[61]:


# Identify top 10 users by total karma
top_users_karma = df.groupby('author_name')['user_total_karma'].sum().sort_values(ascending=False).head(10)

# Plot the top 10 users by total karma
plt.figure(figsize=(10, 6))
top_users_karma.plot(kind='bar')
plt.title('Top 10 Users by Total Karma')
plt.xlabel('Username')
plt.ylabel('Total Karma')
plt.show()


# In[62]:


# Create a new column for the date (ignoring the time part)
df['date'] = df['created_time'].dt.date

# Group by date to count the number of comments
daily_comments = df.groupby('date').size()

# Plot the number of comments over time
plt.figure(figsize=(10, 6))
daily_comments.plot()
plt.title('Number of Comments Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Comments')
plt.show()


# In[63]:


# Group by date and calculate the mean score for each day
daily_score = df.groupby('date')['score'].mean()

# Plot average comment score over time
plt.figure(figsize=(10, 6))
daily_score.plot()
plt.title('Average Comment Score Over Time')
plt.xlabel('Date')
plt.ylabel('Average Score')
plt.show()


# Snetiment Analysis 

# In[64]:


# Install with: pip install vaderSentiment  (see requirements.txt)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# In[65]:


# Sentiment Analysis for Comments
df['comment_sentiment'] = df['cleaned_self_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['comment_sentiment_category'] = df['comment_sentiment'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))


# In[66]:


sentiment_counts = df['comment_sentiment_category'].value_counts()
print(sentiment_counts)
save_table(sentiment_counts, "sentiment_counts")

plt.figure(figsize=(8, 5))
sns.countplot(x='comment_sentiment_category', data=df, palette='viridis')
plt.title('Sentiment Category Distribution')
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
plt.show()


# In[67]:


import plotly.graph_objects as go

# Creating the funnel chart using the sentiment results
sentiment_counts_df = pd.DataFrame({
    'Post_Sentiment': ['Negative', 'Positive', 'Neutral'],
    'Count': [796587, 635028, 377694]
})

fig = go.Figure(go.Funnelarea(
    text=sentiment_counts_df['Post_Sentiment'],
    values=sentiment_counts_df['Count'],
    title={"position": "top center"}
))
fig.update_layout(
    title="Funnel-Chart of Sentiment Distribution | Comments",
    title_x=0.5, width=500, height=400
)
try:
    fig.write_image(os.path.join(FIGURES_DIR, "funnel_sentiment.png"))
    fig.write_html(os.path.join(FIGURES_DIR, "funnel_sentiment.html"))
except Exception as exc:
    print(f"[warn] could not save plotly funnel: {exc}")
fig.show()


# Checking for some positve negetive and neutral sentinments 
# 

# In[68]:


# Filter the DataFrame for positive comments
positive_posts = df[df['comment_sentiment_category'] == 'positive']

# Display a sample of positive posts
print("Sample of Positive Posts:")
positive_posts[['self_text', 'subreddit', 'created_time']].head(10)


# In[69]:


# Filter the DataFrame for negative comments
negative_posts = df[df['comment_sentiment_category'] == 'negative']
print("\nSample of Negative Posts:")
negative_posts[['self_text', 'subreddit', 'created_time']].head(10)



# In[70]:


# Filter the DataFrame for neutral comments
neutral_posts = df[df['comment_sentiment_category'] == 'neutral']
print("\nSample of Neutral Posts:")
neutral_posts[['self_text', 'subreddit', 'created_time']].head(10)


# Weekly Sentiment Trends

# Sentiment Analysis by Subreddit:
# 

# In[71]:


sentiment_by_subreddit = df.groupby('subreddit')['comment_sentiment_category'].value_counts(normalize=True).unstack()
sentiment_by_subreddit.plot(kind='bar', stacked=True, figsize=(15, 7), colormap='viridis')
plt.title('Sentiment Distribution Across Subreddits')
plt.xlabel('Subreddit')
plt.ylabel('Proportion')
plt.legend(title='Sentiment Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[72]:


# Pro-Israel and Pro-Palestine keyword lists
pro_israel_keywords = [
    'idf', 'israel defense', 'right to defend', 'terrorism', 'self-defense',
    'security barrier', 'rocket attacks', 'right to exist', 'two-state solution',
    'peace process', 'settlements', 'annexation', 'jewish homeland', 'zionism',
    'diaspora', 'temple mount', 'zionist', 'netanyahu', 'israeli'
]

pro_palestine_keywords = [
    'palestinian rights', 'occupation', 'freedom', 'apartheid', 'ethnic cleansing',
    'intifada', 'freedom fighters', 'resistance', 'bds', 'boycott', 'gaza blockade',
    'refugees', 'right of return', 'human rights violations', 'UN resolutions',
    'palestine', 'hamas', 'gaza', 'west bank', 'palestinian'
]

# Simplified function to classify stance based on keywords and sentiment
def classify_stance_simple(text, sentiment_score):
    text_lower = text.lower()
    
    # Check for Israel-supporting keywords
    is_israel = any(keyword in text_lower for keyword in pro_israel_keywords)
    
    # Check for Palestine-supporting keywords
    is_palestine = any(keyword in text_lower for keyword in pro_palestine_keywords)
    
    # Classify based on presence of keywords and sentiment
    if is_israel:
        return 'pro-israel'
    elif is_palestine:
        return 'pro-palestine'
    else:
        return 'neutral'

# Apply classification using the simplified function
df['stance_simple'] = df.apply(
    lambda row: classify_stance_simple(row['cleaned_self_text'], row['comment_sentiment']), axis=1
)

# Summary of stance classification
stance_summary_simple = df['stance_simple'].value_counts()
print(stance_summary_simple)
save_table(stance_summary_simple, "stance_summary")


# In[73]:


# Group by 'subreddit' and 'stance_simple' to count the number of comments
support_counts = df.groupby(['subreddit', 'stance_simple']).size().unstack(fill_value=0)
save_table(support_counts.reset_index(), "stance_by_subreddit")

# Display the support counts
support_counts


# In[74]:


# Calculate the proportion of each stance in each subreddit
support_distribution = support_counts.div(support_counts.sum(axis=1), axis=0)

# Plot the support distribution as a stacked bar chart
support_distribution.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='viridis')
plt.title('Support Distribution Across Subreddits')
plt.xlabel('Subreddit')
plt.ylabel('Proportion of Support')
plt.legend(title='Stance', bbox_to_anchor=(1, 1))
plt.show()


# In[75]:


# Count the number of unique users per subreddit
unique_users_per_subreddit = df.groupby('subreddit')['author_name'].nunique().reset_index()
unique_users_per_subreddit.columns = ['subreddit', 'unique_users']
save_table(unique_users_per_subreddit, "unique_users_per_subreddit")

# Display the result
print(unique_users_per_subreddit)


# In[76]:


# Group by 'author_name' to calculate total number of comments and total score
user_activity = df.groupby('author_name').agg({
    'score': 'sum',  # Total score received by the user
    'comment_id': 'count'  # Total number of comments made by the user
}).reset_index()

# Rename columns for clarity
user_activity.columns = ['author_name', 'total_score', 'total_comments']

# Sort by total score and total comments to identify top users
top_users = user_activity.sort_values(by=['total_score', 'total_comments'], ascending=False).head(10)

# Display top users
top_users


# In[77]:


# Calculate the dominant stance for each user
user_stance_consistency = df.groupby('author_name')['stance_simple'].agg(lambda x: x.mode()[0]).reset_index()

# Count the number of users with each dominant stance
stance_consistency_summary = user_stance_consistency['stance_simple'].value_counts()
print(stance_consistency_summary)


# In[78]:


plt.figure(figsize=(12, 6))
sns.barplot(x='unique_users', y='subreddit', data=unique_users_per_subreddit.sort_values(by='unique_users', ascending=False), palette='Blues_d')
plt.title('Number of Unique Users per Subreddit')
plt.xlabel('Unique Users')
plt.ylabel('Subreddit')
plt.show()


# In[79]:


# Sentiment Analysis for Post Title
df['post_title_sentiment'] = df['cleaned_post_title'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['post_title_sentiment_category'] = df['post_title_sentiment'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))


# In[80]:


plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 3)
sns.countplot(x='post_title_sentiment_category', data=df, palette='coolwarm')
plt.title('Post Title Sentiment Distribution')
plt.xlabel('Sentiment Category')
plt.ylabel('Count')


# In[81]:


# Display a few sample positive posts
positive_posts = df[df['post_title_sentiment_category'] == 'positive'][['post_title', 'post_self_text']].sample(5, random_state=42)
print("Positive Posts:")
positive_posts


# In[82]:


# Display a few sample negative posts
negative_posts = df[df['post_title_sentiment_category'] == 'negative'][['post_title', 'post_self_text']].sample(5, random_state=42)
print("\nNegative Posts:")
negative_posts


# In[83]:


# Display a few sample neutral posts
neutral_posts = df[df['post_title_sentiment_category'] == 'neutral'][['post_title', 'post_self_text']].sample(5, random_state=42)
print("\nNeutral Posts:")
neutral_posts


# Calculating TF-IDf scores 

# In[84]:


from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization for frequent terms extraction
tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_self_text'])

# Extract top terms and their TF-IDF scores
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1
tfidf_scores_df = pd.DataFrame(list(zip(tfidf_feature_names, tfidf_scores)), columns=['term', 'score'])
tfidf_scores_df = tfidf_scores_df.sort_values(by='score', ascending=False)
save_table(tfidf_scores_df, "tfidf_top_terms_comments")

# Display top 20 terms
print("Top 20 Terms by TF-IDF Score:")
print(tfidf_scores_df.head(20))

# Visualization: Bar plot of top terms
plt.figure(figsize=(12, 6))
sns.barplot(x='score', y='term', data=tfidf_scores_df.head(20), palette='viridis')
plt.title('Top 20 Terms by TF-IDF Score')
plt.xlabel('TF-IDF Score')
plt.ylabel('Term')
plt.show()


# In[85]:


# 1. Sentiment Distribution Over Time
df['created_week'] = df['created_time'].dt.to_period('W').dt.to_timestamp()
sentiment_over_time = df.groupby('created_week')['comment_sentiment'].mean()

plt.figure(figsize=(14, 6))
sns.lineplot(x=sentiment_over_time.index, y=sentiment_over_time, marker='o')
plt.title('Average Sentiment Over Time')
plt.xlabel('Week')
plt.ylabel('Average Sentiment Score')
plt.xticks(rotation=45)
plt.show()


# In[86]:


# 2. Stance Distribution
stance_counts = df['stance_simple'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(stance_counts, labels=stance_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff6666','#99ff99'])
plt.title('Stance Distrlibution (Pro-Israel, Pro-Palestine, Neutral)')
plt.show()


# In[87]:


# 4. Heatmap for Subreddit Sentiment
subreddit_sentiment = df.pivot_table(index='subreddit', columns='stance_simple', values='comment_sentiment', aggfunc='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(subreddit_sentiment, cmap='coolwarm', annot=True)
plt.title('Average Sentiment by Subreddit and Stance')
plt.show()


# In[88]:


# Count instances for Pro-Israel and Pro-Palestine
pro_israel_count = df['stance_simple'].value_counts().get('pro-israel', 0)
pro_palestine_count = df['stance_simple'].value_counts().get('pro-palestine', 0)
neutral_count = df['stance_simple'].value_counts().get('neutral', 0)

# Bar Plot
plt.figure(figsize=(8, 6))
sns.barplot(x=['Pro-Israel', 'Pro-Palestine', 'Neutral'], y=[pro_israel_count, pro_palestine_count, neutral_count], palette='coolwarm')
plt.title('Distribution of Stance in Comments')
plt.ylabel('Number of Comments')
plt.show()


# In[89]:


# TF-IDF Vectorization for 'cleaned_post_title'
tfidf_matrix_post_title = tfidf_vectorizer.fit_transform(df['cleaned_post_title'].fillna(''))

# Extract top terms and their TF-IDF scores
tfidf_feature_names_post_title = tfidf_vectorizer.get_feature_names_out()
tfidf_scores_post_title = tfidf_matrix_post_title.sum(axis=0).A1
tfidf_scores_post_title_df = pd.DataFrame(
    list(zip(tfidf_feature_names_post_title, tfidf_scores_post_title)), 
    columns=['term', 'score']
).sort_values(by='score', ascending=False)

# Display top 20 terms
print("Top 20 Terms by TF-IDF Score for 'cleaned_post_title':")
print(tfidf_scores_post_title_df.head(20))

# Visualization: Bar plot of top terms for 'cleaned_post_title'
plt.figure(figsize=(12, 6))
sns.barplot(x='score', y='term', data=tfidf_scores_post_title_df.head(20), palette='plasma')
plt.title('Top 20 Terms by TF-IDF Score for cleaned_post_title')
plt.xlabel('TF-IDF Score')
plt.ylabel('Term')
plt.show()


# In[90]:


from collections import Counter
from nltk import bigrams
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample a smaller subset of data to avoid memory issues (e.g., 10,000 rows)
sample_size = 1000000 # Adjust this number based on your system's capacity
df_sampled = df['cleaned_self_text'].sample(n=sample_size, random_state=42)

# Define a generator function to yield bigrams from a series of texts
def generate_bigrams(text_series):
    for text in text_series:
        yield from bigrams(text.split())

# Use the generator to create bigrams and count them
bigram_counts = Counter(generate_bigrams(df_sampled))

# Convert the top 20 most common bigrams to a DataFrame for visualization
bigram_df = pd.DataFrame(bigram_counts.most_common(20), columns=['bigram', 'count'])
_bigram_export = bigram_df.copy()
_bigram_export['bigram'] = _bigram_export['bigram'].apply(lambda b: ' '.join(b))
save_table(_bigram_export, "top_bigrams")

# Visualization: Bar plot of top 20 bigrams
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y=[' '.join(bigram) for bigram in bigram_df['bigram']], data=bigram_df, palette='viridis')
plt.title('Top 20 Bigrams')
plt.xlabel('Count')
plt.ylabel('Bigram')
plt.show()


# In[91]:


# List of top 20 key terms based on TF-IDF scores
key_terms = tfidf_scores_df.head(20)['term'].tolist()

# Analyze context and sentiment of key terms
for term in key_terms:
    # Subset of comments containing the term
    subset = df[df['cleaned_self_text'].str.contains(term, case=False, na=False)]
    
    # Calculate the average sentiment for the subset
    avg_sentiment = subset['comment_sentiment'].mean()
    
    # Print the results
    print(f"Average sentiment for comments containing '{term}': {avg_sentiment:.2f}")


# Emotion detection 

# In[92]:


from nrclex import NRCLex

# Emotion Detection Function
def detect_emotions(text):
    emotion = NRCLex(text)
    return emotion.raw_emotion_scores

# Apply emotion detection to cleaned text
df['emotion_scores'] = df['cleaned_self_text'].apply(detect_emotions)

# Create separate columns for each emotion
emotion_columns = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 'positive', 'negative', 'sadness', 'disgust', 'joy']
for col in emotion_columns:
    df[col] = df['emotion_scores'].apply(lambda x: x.get(col, 0))

# Visualize the average emotion scores
avg_emotions = df[emotion_columns].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_emotions.index, y=avg_emotions.values, palette='coolwarm')
plt.title('Average Emotion Scores for Comments')
plt.xlabel('Emotion')
plt.ylabel('Average Score')
plt.show()


# In[93]:


# Emotion Distribution for Key Terms
key_terms = tfidf_scores_df.head(20)['term'].tolist()  # Use the top 20 key terms from TF-IDF
for term in key_terms:
    subset = df[df['cleaned_self_text'].str.contains(term, na=False)]
    avg_emotions_subset = subset[emotion_columns].mean()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=avg_emotions_subset.index, y=avg_emotions_subset.values, palette='Spectral')
    plt.title(f'Emotion Distribution for Comments Containing "{term}"')
    plt.xlabel('Emotion')
    plt.ylabel('Average Score')
    plt.show()


# In[94]:


# Emotion Patterns Across Time
emotion_time_series = df.groupby('date')[emotion_columns].mean()

plt.figure(figsize=(16, 8))
for col in emotion_columns:
    plt.plot(emotion_time_series.index, emotion_time_series[col], label=col)
plt.title('Emotion Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Average Emotion Score')
plt.legend()
plt.show()

# Emotion Patterns Across Subreddits
top_subreddits = df['subreddit'].value_counts().head(5).index
emotion_by_subreddit = df[df['subreddit'].isin(top_subreddits)].groupby('subreddit')[emotion_columns].mean()
save_table(emotion_by_subreddit.reset_index(), "emotion_by_subreddit")

plt.figure(figsize=(14, 8))
sns.heatmap(emotion_by_subreddit.T, annot=True, cmap='coolwarm')
plt.title('Emotion Scores Across Top Subreddits')
plt.xlabel('Subreddit')
plt.ylabel('Emotion')
plt.show()


# Applying LDA

# In[120]:


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

# Function to perform LDA and visualize topics without SVD
def perform_lda_optimized(df, text_column, num_topics=3, num_words=20):
    # Sample a subset of data for faster computation
    sampled_df = df.sample(n=100000, random_state=42)  # Adjust n as needed
    
    # Vectorize the text data
    vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
    dtm = vectorizer.fit_transform(sampled_df[text_column])
    
    # Apply LDA directly on the dtm
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42, max_iter=10, n_jobs=-1)
    lda_model.fit(dtm)
    
    # Extract topics
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics[f"Topic {idx + 1}"] = top_words
        
        # Word Cloud for each topic
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate(" ".join(top_words))
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for Topic {idx + 1} in {text_column}")
        plt.show()

    # Persist the topic-word table for this text column
    save_table(
        pd.DataFrame([(t, ", ".join(words)) for t, words in topics.items()],
                     columns=["topic", "top_words"]),
        f"lda_topics_{text_column}",
    )

    # Return the LDA model and feature names for bar graph visualization
    return lda_model, feature_names

# LDA on cleaned_self_text
lda_model_self_text, feature_names_self_text = perform_lda_optimized(df, 'cleaned_self_text', num_topics=5)


# In[96]:


# Topic labeling based on word clouds and dominant words
topics = {
    "Topic 1": "Geopolitical Conflict",
    "Topic 2": "Military Actions and Casualties",
    "Topic 3": "International Relations",
    "Topic 4": "Religious and Cultural Issues",
    "Topic 5": "Media and Public Reactions"
}


# In[97]:


# Function to visualize LDA topics as bar graphs
def visualize_lda_topics(lda_model, feature_names, num_topics=5, num_words=10):
    for idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        word_scores = [topic[i] for i in topic.argsort()[:-num_words - 1:-1]]
        
        # Convert top words and their scores to a DataFrame
        topic_df = pd.DataFrame(list(zip(top_words, word_scores)), columns=['word', 'score'])
        
        # Plotting the top words for each topic
        plt.figure(figsize=(10, 6))
        sns.barplot(x='score', y='word', data=topic_df, palette='viridis')
        plt.title(f'Top {num_words} Words for Topic {idx + 1}')
        plt.xlabel('Score')
        plt.ylabel('Word')
        plt.show()

# Visualize topics as bar graphs for cleaned_self_text
visualize_lda_topics(lda_model_self_text, feature_names_self_text, num_topics=5, num_words=10)


# In[98]:


# LDA on cleaned_post_title
lda_model_post_title, feature_names_post_title = perform_lda_optimized(df, 'cleaned_post_title', num_topics=5)
# Visualize topics as bar graphs for cleaned_post_title
visualize_lda_topics(lda_model_post_title, feature_names_post_title, num_topics=5, num_words=10)


# In[99]:


# Define keywords for detecting misinformation or propaganda
misinformation_keywords = [
    'fake news', 'hoax', 'false flag', 'conspiracy', 'deep state', 'propaganda',
    'misinformation', 'disinformation', 'biased', 'fake', 'hoax', 'lies', 'false'
]
# Function to detect misinformation-related content
def detect_misinformation(text):
    for word in misinformation_keywords:
        if word in text.lower():
            return 1
    return 0

# Apply misinformation detection
df['misinformation_flag'] = df['cleaned_self_text'].apply(detect_misinformation)

# Check the number of misinformation-flagged comments
misinformation_count = df['misinformation_flag'].sum()
print(f"Total Misinformation-Flagged Comments: {misinformation_count}")
save_table(
    pd.DataFrame({
        "metric": ["misinformation_flagged", "non_flagged"],
        "count": [int(misinformation_count), int(len(df) - misinformation_count)]
    }),
    "misinformation_counts",
)


# In[100]:


# Calculate average sentiment for misinformation-flagged and non-misinformation comments
misinformation_sentiment = df[df['misinformation_flag'] == 1]['comment_sentiment'].mean()
non_misinformation_sentiment = df[df['misinformation_flag'] == 0]['comment_sentiment'].mean()

# Print results
print(f"Avg. Sentiment of Misinformation-flagged Comments: {misinformation_sentiment}")
print(f"Avg. Sentiment of Non-Misinformation Comments: {non_misinformation_sentiment}")


# In[101]:


# Visualization of sentiment distribution for misinformation vs. non-misinformation comments
plt.figure(figsize=(8, 6))
sns.boxplot(x='misinformation_flag', y='comment_sentiment', data=df, palette='coolwarm')
plt.title('Sentiment Distribution of Misinformation vs. Non-Misinformation Comments')
plt.xlabel('Misinformation Flag (0 = Non-Misinformation, 1 = Misinformation)')
plt.ylabel('Sentiment Score')
plt.show()


# In[102]:


# Summary statistics for sentiment scores
summary_stats = df.groupby('misinformation_flag')['comment_sentiment'].describe()
summary_stats


# In[103]:


# Using 'comment_sentiment' for sentiment and 'score' for upvotes/downs
user_behavior = df.groupby('author_name').agg({
    'self_text': 'count',  # Count of comments per user
    'comment_sentiment': 'mean',  # Average sentiment per user
    'score': 'sum'  # Total score per user (you can use 'ups' if available)
}).reset_index()

user_behavior.columns = ['author_name', 'comment_count', 'average_sentiment', 'total_score']

# Top 10 users by comment count
top_users = user_behavior.sort_values(by='comment_count', ascending=False).head(10)
print("Top 10 Users by Comment Count:")
print(top_users)
# Visualization of top users by comment count
plt.figure(figsize=(10, 6))
sns.barplot(x='comment_count', y='author_name', data=top_users, palette='coolwarm')
plt.title('Top 10 Most Active Users by Comment Count')
plt.xlabel('Comment Count')
plt.ylabel('User')
plt.show()

# Visualization of sentiment distribution for top users
plt.figure(figsize=(10, 6))
sns.barplot(x='average_sentiment', y='author_name', data=top_users, palette='coolwarm')
plt.title('Average Sentiment of Top 10 Most Active Users')
plt.xlabel('Average Sentiment')
plt.ylabel('User')
plt.show()


# In[104]:


# Calculate average replies and upvotes for each user
engagement_df = df.groupby('author_name').agg({
    'score': 'mean',  # Average upvotes
    'self_text': 'count'  # Comment count
}).reset_index()

# Merge with top users data
top_users_engagement = top_users.merge(engagement_df, on='author_name', how='left')

# Plot engagement metrics
plt.figure(figsize=(10, 6))
sns.barplot(x='score', y='author_name', data=top_users_engagement, palette='coolwarm')
plt.title('Average Score for Top 10 Users')
plt.xlabel('Average Score')
plt.ylabel('User')
plt.show()


# In[105]:


df.columns


# In[109]:


# Aggregate sentiment by weekly average
weekly_sentiment_trends = df.groupby(df['created_time'].dt.to_period('W')).agg({'comment_sentiment': 'mean'}).reset_index()
weekly_sentiment_trends['created_time'] = weekly_sentiment_trends['created_time'].dt.start_time  # Ensure it's in the correct format
weekly_sentiment_trends.columns = ['ds', 'y']  # Prophet requires columns 'ds' and 'y'

print(weekly_sentiment_trends.head())


# In[110]:


from prophet import Prophet

# Initialize and fit the Prophet model for weekly data
prophet_model_weekly = Prophet()
prophet_model_weekly.fit(weekly_sentiment_trends)

# Create a dataframe to hold predictions for the next 12 weeks
future_weekly_dates = prophet_model_weekly.make_future_dataframe(periods=12, freq='W')
forecast_weekly = prophet_model_weekly.predict(future_weekly_dates)

# Plot the forecast for weekly sentiment trends
fig_weekly = prophet_model_weekly.plot(forecast_weekly)
plt.title('Weekly Sentiment Trend Forecast')
plt.xlabel('Date')
plt.ylabel('Average Weekly Sentiment')
plt.show()

# Plot components (trend, weekly seasonality)
fig_weekly_components = prophet_model_weekly.plot_components(forecast_weekly)
plt.show()


# In[111]:


# Group by subreddit and sentiment category
subreddit_sentiment_distribution = df.groupby(['subreddit', 'comment_sentiment_category']).size().unstack(fill_value=0)

# Calculate sentiment proportions per subreddit
subreddit_sentiment_distribution_normalized = subreddit_sentiment_distribution.div(subreddit_sentiment_distribution.sum(axis=1), axis=0)

# Plot sentiment distribution as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(subreddit_sentiment_distribution_normalized, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Sentiment Distribution across Subreddits')
plt.xlabel('Sentiment Category')
plt.ylabel('Subreddit')
plt.show()


# In[113]:


from scipy.stats import chi2_contingency
# Perform Chi-square test
chi2, p, dof, expected = chi2_contingency(subreddit_sentiment_distribution)

print(f"Chi-square Statistic: {chi2}")
print(f"P-value: {p}")
if p < 0.05:
    print("There is a significant difference in sentiment distribution across subreddits.")
else:
    print("There is no significant difference in sentiment distribution across subreddits.")
save_table(
    pd.DataFrame({
        "statistic": ["chi2", "p_value", "dof"],
        "value": [chi2, p, dof],
    }),
    "chi_square_subreddit_sentiment",
)


# In[114]:


# Checking distribution of controversial comments
controversial_counts = df['controversiality'].value_counts()
print("Controversiality Counts:\n", controversial_counts)
save_table(controversial_counts, "controversiality_counts")

# Visualization of controversial vs non-controversial comments
plt.figure(figsize=(6,4))
sns.countplot(x='controversiality', data=df, palette='coolwarm')
plt.title('Distribution of Controversial vs Non-Controversial Comments')
plt.xlabel('Controversiality (0 = Not Controversial, 1 = Controversial)')
plt.ylabel('Count')
plt.show()

# Compare sentiment of controversial vs non-controversial comments
plt.figure(figsize=(10, 6))
sns.boxplot(x='controversiality', y='comment_sentiment', data=df, palette='coolwarm')
plt.title('Sentiment Distribution of Controversial vs Non-Controversial Comments')
plt.xlabel('Controversiality (0 = Not Controversial, 1 = Controversial)')
plt.ylabel('Comment Sentiment')
plt.show()

# Checking average sentiment for controversial vs non-controversial
avg_sentiment_by_controversial = df.groupby('controversiality')['comment_sentiment'].mean()
print("Average Sentiment by Controversiality:\n", avg_sentiment_by_controversial)
save_table(avg_sentiment_by_controversial, "avg_sentiment_by_controversial")


# In[117]:


# Get the top 5 subreddits by comment count
top_5_subreddits = df['subreddit'].value_counts().head(5).index.tolist()

# Filter data for top 5 subreddits
df_top_5_subreddits = df[df['subreddit'].isin(top_5_subreddits)]

# Plot sentiment distribution for top subreddits
plt.figure(figsize=(12, 6))
sns.countplot(x='subreddit', hue='comment_sentiment_category', data=df_top_5_subreddits, palette='coolwarm')
plt.title('Sentiment Distribution for Top 5 Subreddits')
plt.xlabel('Subreddit')
plt.ylabel('Count')
plt.legend(title='Sentiment', loc='upper right')
plt.tight_layout()
plt.show()


# In[118]:


# Grouping by subreddit and week to track sentiment over time
weekly_sentiment_subreddit = df_top_5_subreddits.groupby(['created_week', 'subreddit'])['comment_sentiment'].mean().reset_index()

# Plotting sentiment evolution for each subreddit
plt.figure(figsize=(12, 8))
sns.lineplot(x='created_week', y='comment_sentiment', hue='subreddit', data=weekly_sentiment_subreddit, marker='o', palette='tab10')
plt.title('Sentiment Evolution Over Time for Top 5 Subreddits')
plt.xlabel('Week')
plt.ylabel('Average Sentiment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

