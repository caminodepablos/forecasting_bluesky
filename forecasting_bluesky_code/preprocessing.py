import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

def preprocessing_news_df(df, subject = 'None'):
    '''
    Receives: df, subject name if not in original dataframe. 
    Process:
    - Date column to datetime
    - Date filtering: 2023-05-08 - 2025-04-07
    - Index reseting
    - Values sorted by date
    - Creates subject column if given
    '''
    
    # Date column to Dtype = 'datetime'
    df['date'] = pd.to_datetime(df['date'])

    # Date filtering: 2023-05-08 - 2025-04-07
    df = df[(df['date'] >= '2023-05-08') & (df['date'] <= '2025-04-07')]

    # Sort values by 'date'
    df = df.sort_values(by='date')
    
    # Reset index 
    df = df.reset_index(drop=True)

    # Creates 'subject' column if given
    if subject != 'None':
        df['subject'] = subject

    # Shows df info
    print(df.info())

    # Returns preprocessed dataframe
    return df

def preprocessing_bsky_df(df):
    '''
    Receives bsky stats dataset and returns the df preprocessed.
    Process:
    - Date to datetime
    - New_users column
    - Drops non-used activity cols
    '''
    # 'date' column to 'datetime' dtype
    df['date'] = pd.to_datetime(df['date'])
    
    # 'new_users' column 
    df['new_users'] = df['users'].diff()
    df['new_users'] = df['new_users'].fillna(0)
    
    # column dropping
    df.drop(columns = (['num_likers', 'num_first_time_posters', 'num_posters', 'num_posts_with_images', 'num_images_with_alt_text', 'num_followers', 'num_blockers']), inplace=True)
    
    # column sorting
    df = df[['date', 'year', 'month', 'day', 'users', 'new_users', 'num_likes', 'num_posts', 'num_images', 'num_follows', 'num_blocks']]

    return df

def basic_proc_final_dataset(df):
    '''
    Receives final bsky & news dataset and returns:
    - NaN = ''
    - Date column = datetime
    '''
    df = df.fillna('')
    df['date'] = pd.to_datetime(df['date'])
    df.rename(columns = {'users':'tot_users'}, inplace = True)
    # set date column as index
    df = df.set_index('date')
    return df

def text_cleaning(text):
    '''
    Basic text cleaning (English):
    - NaN: ''
    - Lowercase
    - Eliminate punctuations
    - Space cleaning
    '''
    if pd.isnull(text):
        return ""
    # Lowercase
    text = text.lower()
    # Eliminate punctuations
    text = re.sub(r"[^a-záéíóúüñ\s]", "", text)
    # Space cleaning
    text = re.sub(r"\s+", " ", text).strip()
    return text

def activity_score_pca(df, activity_cols = ['num_likes', 'num_posts', 'num_images', 'num_follows', 'num_blocks']):
    '''
    Receive full dataframe and columns list with bsky activity information (optional).
    Applies StandardScaler() and PCA().
    PCA: Principal Components Analysis.
    See documentation: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    '''
    
    # mean 0, std 1
    scaler = StandardScaler()
    norm_activity = scaler.fit_transform(df[activity_cols])
    
    # PCA
    pca = PCA(n_components=1)
    df['activity_score_pca'] = pca.fit_transform(norm_activity)
    
    # Activity cols dropping
    df.drop(columns = (activity_cols), inplace=True)

    return df

def tf_idf(df, max_features=1000, ngram_range=(1,2), max_df=0.99, norm='l2'):
    '''
    Receives full dataframe (must have date column) and some optionals params.
    Preprocess and applies TF-IDF to all text columns in dataframe.
    Returns:
    - New dataframe with date, num features and tf-idf vectors
    - TF-IDF Matrix
    '''

    df = df.reset_index(drop=False)
    
    # We save the 'date' column as Series for future reference
    date_col = df['date']
    
    # Numeric & Categoric
    num_ft = df._get_numeric_data().columns
    cat_ft = df.drop(columns=list(num_ft)+['date']).columns
    
    # Basic text cleaning (spaces, punctuation and NaN)
    for column in cat_ft:
        df[column] = df[column].apply(text_cleaning)
        
    # Text join
    df['full_text'] = df[cat_ft].agg(" ".join, axis=1)
    
    # Pipeline
    vectorizer = Pipeline([
        ('bow', CountVectorizer(max_features=max_features, stop_words='english', ngram_range=ngram_range, max_df=max_df)), # Bag-of-Words
        ('tfidf', TfidfTransformer(norm=norm)) # TF-IDF
    ])
    
    tfidf_matrix = vectorizer.fit_transform(df['full_text'])
    
    # To DataFrame 
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.named_steps['bow'].get_feature_names_out(),
        index=df.index
    )
    
    # Num features concat
    df_final = pd.concat([
        date_col,
        df[num_ft],
        tfidf_df
    ], axis=1)

    df_final = df_final.set_index('date')

    return df_final, tfidf_matrix

def temp_columns(df, columns = ['tot_users', 'new_users', 'activity_score_pca'], window = 7):
    '''
    Receives full dataframe (must have date colum) and list of columns to create (optional). Returns full df with new temp columns:
    - Lags (yesterday data for users, new users and activity score)
    - Rolling Mean (7 day rolling mean data for new users and activity score)
    - Diff (diff today vs yesterday for new users and activity score)
    - NaN = 0
    '''
    
    # Lags
    df['users_lag1'] = df['tot_users'].shift(1)
    df['new_users_lag1'] = df['new_users'].shift(1)
    df['activity_lag1'] = df['activity_score_pca'].shift(1)
    
    # Rolling mean
    df['new_users_ma7'] = df['new_users'].rolling(window=window).mean()
    df['activity_ma7'] = df['activity_score_pca'].rolling(window=window).mean()
    
    # Diff
    df['new_users_diff'] = df['new_users'].diff()
    df['activity_diff'] = df['activity_score_pca'].diff()

    # Fills NaN with 0
    df = df.fillna(0)

    return df

def stemming(headline):
    '''
    Headline stemming. Receives a phrase and returns the phrase stemmed word by word
    '''

    # Objeto de Stemming
    stemmer = PorterStemmer()
    
    # Tokenizing the headline
    headline = nltk.tokenize.RegexpTokenizer("[\\w]+").tokenize(headline)
    
    # Word over 2 characters list
    headline = [word for word in headline if len(word)>=3]

    # Stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    
    # Stopwords
    headline = [word for word in headline if word not in stopwords]

    # STEMMING
    headline = [stemmer.stem(word) for word in headline]
    
    # Join each word in a phrase
    headline = " ".join(headline)
    
    return headline

def lemmatizer(headline):
    '''
    Headline lemmatizer. Receiveis a phrase and returns the phrase lemmatized word by word
    '''

    # Objeto de Lemmatizing
    wordnet_lemmatizer = WordNetLemmatizer()
    
    # Tokenizing the headline
    headline = nltk.tokenize.RegexpTokenizer("[\\w]+").tokenize(headline)
    
    # Word over 2 characters list
    headline = [word for word in headline if len(word)>=3]

    # Stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    
    # Stopwords
    headline = [word for word in headline if word not in stopwords]

    # LEMMATIZING
    new_tokens=[] # lemmatized
    tag_dict = {"J": wordnet.ADJ, # adjective (a)
                "N": wordnet.NOUN, # name (n)
                "V": wordnet.VERB, # verb (v)
                "S": wordnet.ADJ_SAT, # satellite adjectives (s)
                "R": wordnet.ADV} # adverb (r)

    # Tuples list (ex: [('Laura', 'NNP'), ('teachs', 'VBZ'), ('math', 'NN')])
    words_and_pos = nltk.pos_tag(headline)

    for items in words_and_pos:
        pos = tag_dict.get(items[1][0], wordnet.NOUN) # if not found = name
        new_tokens.append(wordnet_lemmatizer.lemmatize(items[0], pos))
    
    # Join each word in a phrase
    headline_lem = " ".join(new_tokens)
    
    return headline_lem

def apply_stem_lemm(df, function):
    '''
    Applies either stemming or lemmatizing to every text of a dataframe. 
    Returns the dataframe complete.
    Function: "stem" for stemming / "lem" for lemmatizing
    '''
    
    # Categoric & Numeric Features
    num_ft = df._get_numeric_data().columns
    cat_ft = df.drop(columns=num_ft).columns

    if function == 'stem':
        for cat in cat_ft:
            df[cat] = df[cat].apply(stemming)
    
    elif function == 'lem':
        for cat in cat_ft:
            df[cat] = df[cat].apply(lemmatizer)

    return df

def categorize_quantile(df, var, quantile_num = [0.25, 0.5, 0.75]):
    '''
    Receives a dataframe column (var) and a list of 3 quantile (optional).
    Returns new column with categorization applied
    '''

    quantile = df[var].quantile(quantile_num)
    
    def categorize_score(score):
        if score <= quantile[quantile_num[0]]:
            return 0 # low
        elif score <= quantile[quantile_num[1]]:
            return 1 # medium
        elif score <= quantile[quantile_num[2]]:
            return 2 # high
        else:
            return 3 # very high
    
    return df[var].apply(categorize_score)

def categorize_new_user_threshold(df, var = 'new_users', threshold = 14):
    '''
    Receives full dataframe, a var name (optional) and a threshold (default = 14 days).
    Return df with new column: new_users_above_avg.
    Applies a threshold using 2 weeks previus mean.
    '''
    
    # Rolling mean 14 previus dais (except current day)
    df['mean_prev_threshold'] = df['new_users'].shift(1).rolling(window=threshold).mean()
    
    # Target var
    df['new_users_above_avg'] = (df['new_users'] > df['mean_prev_threshold']).astype(int)

    # Fill first NaN values with 0
    df['new_users_above_avg'] = df['new_users_above_avg'].fillna(0)

    return df

def binary_news_categorization(df):
    '''
    News binary categorization: 1 has headline / 0 doesn't have headline
    '''
    # Numeric & Categoric Features
    num_ft = df._get_numeric_data().columns
    cat_ft = df.drop(columns=num_ft).columns
    
    for col in cat_ft:
        df[col] = df[col].apply(lambda x: 1 if x != '' else 0)
    
    return df

def generate_weekly_df(df, period = 'W'):
    '''
    Receives a daily dataframe and a preferred period (Weekly by default). Returns the aggregated dataframe:
    - Activity columns: mean
    - Total Users: last period record
    - New users: difference week-to-week
    - News: text aggregated by period
    '''

    # Activity cols mean
    activity_cols = ['num_likes', 'num_posts', 'num_images', 'num_follows', 'num_blocks']
    activity = df[activity_cols].resample(period).mean()

    # Weekly total users
    users = df[['tot_users']].resample(period).last()

    # Aggregated news
    num_ft = df._get_numeric_data().columns
    cat_ft = df.drop(columns=num_ft).columns
    news = df[cat_ft].resample(period).agg(lambda x: ' '.join(x.dropna()))

    # Concat
    new_df = pd.concat([news, activity, users], axis=1)

    # New_users
    new_df['new_users'] = new_df['tot_users'].diff()
    new_df['new_users'] = new_df['new_users'].fillna(0)

    return new_df

def standard_scaler_num(df):
    '''
    Applies StandardScaler() to all numeric columns for a given DataFrame.
    '''
    # Numeric & Categoric Features
    num_ft = df._get_numeric_data().columns
    scaler = StandardScaler()
    df[num_ft] = scaler.fit_transform(df[num_ft])
    
    return df

def get_top_label(headline, classifier):
    """
    Classifies a headline using a zero-shot-classification model.
    Returns the label with the highest score.
    """
    candidate_labels = [
        "politics", "elections", "technology", "artificial intelligence", "science",
        "health", "finance", "environment", "war", "protests", "sports", "entertainment",
        "videogames", "crime", "education", "international relations", "pandemics", "immigration"
    ]
    
    if isinstance(headline, str) and headline.strip():
        result = classifier(headline, candidate_labels)
        top_index = result['scores'].index(max(result['scores']))
        return result['labels'][top_index]
    
    return ''  # Return empty string if input is empty


def zero_shot_classification(df, text_column):
    """
    Applies zero-shot classification to a text column in the dataframe.
    Adds a new column 'subject' with the predicted label.
    """
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Apply classifier row by row
    df['subject'] = df[text_column].apply(lambda x: get_top_label(x, classifier))
    
    return df