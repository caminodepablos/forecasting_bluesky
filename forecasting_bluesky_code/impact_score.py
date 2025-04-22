import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from transformers import pipeline
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline, FeatureUnion
import joblib


class SentimentScoreExtractor(BaseEstimator, TransformerMixin):
    '''
    Calculates a Sentiment-Analysis-based score (-1 to 1) for each text field in the DataFrame.
    Uses DistilBERT for sentiment analysis.
    Score: -1: Negative, 0: Neutral, 1: Positive
    '''
    
    def __init__(self, exclude_columns=None):
        '''
        Exclude_columns = None by default
        
        Works with DistilBERT. 
        Documentation: https://huggingface.co/docs/transformers/model_doc/distilbert
        Model = "distilbert-base-uncased-finetuned-sst-2-english"
        '''
        self.exclude_columns = exclude_columns or []

    def fit(self, X, y=None):

        # Model run
        self.classifier = pipeline(
            task="text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        return self

    def transform(self, X):
        X_ = X.copy()

        # Columns Selection
        text_columns = [col for col in X_.columns if col not in self.exclude_columns]
        score_df = pd.DataFrame(index=X_.index)

        for col in text_columns:
            new_col_name = f'{col}_sentiment_score'
            score_df[new_col_name] = X_[col].apply(lambda x: self._get_sentiment_score(x))
        
        return score_df

    def _get_sentiment_score(self, text):
        if isinstance(text, str) and text.strip() != '':
            result = self.classifier(text)
            label = result[0]['label']
            score = result[0]['score']
            return score if label == 'POSITIVE' else -score
        return 0


class CategoryImpactExtractor(BaseEstimator, TransformerMixin):
    '''
    Calculates a Category-based score (-1 to 1) for each text field in the DataFrame.
    Uses predefined category weights.
    If the category is new, score = 0.
    '''
    def __init__(self, exclude_columns=None):
        '''
        Exclude_columns = None by default
        
        Check out category weights with .category_weights
        '''
        self.exclude_columns = exclude_columns or []

        # Category Weights
        self.category_weights = {
            'artificial intelligence': -0.33,
            'crime': 0.1,
            'education': -0.8,
            'elections': 1.0,
            'entertainment': -0.22,
            'environment': -0.44,
            'finance': 0.52,
            'health': 0.62,
            'immigration': -0.8,
            'international relations': -0.47,
            'pandemics': -0.49,
            'politics': 0.06,
            'protests': -0.35,
            'science': -0.38,
            'sports': -0.81,
            'technology': -0.38,
            'videogames': -1.0,
            'war': 0.04
        }

    def fit(self, X, y=None):
        # Nothing to learn
        return self

    def transform(self, X):
        X_ = X.copy()

        # Columns selection (all if not excluded)
        text_columns = [col for col in X_.columns if col not in self.exclude_columns]
        score_df = pd.DataFrame(index=X_.index)

        for col in text_columns:
            score = self.category_weights.get(col, 0)
            score_col_name = f'{col}_category_score'
            score_df[score_col_name] = score

        return score_df

class NERScoreExtractor(BaseEstimator, TransformerMixin):
    '''
    Calculates a NER-based score (-1 to 1) for each text field in the DataFrame.
    Uses SpaCy and predefined entity weights.
    '''

    def __init__(self, exclude_columns=None, use_whitelist=True):
        '''
        exclude_columns = None by default
        use_whitelist = if True, entities in the whitelist are counted +0.5
        
        Works with SpaCy. 
        Documentation: https://spacy.io/
        Model = en_core_web_sm"
        
        Check out entity weights with .entity_weights
        Check out whitelist with .whitelist
        '''
        
        self.exclude_columns = exclude_columns or []
        self.use_whitelist = use_whitelist

        self.entity_weights = {
            'CARDINAL': -0.69, 'DATE': -0.64, 'EVENT': -0.93, 'FAC': -0.8,
            'GPE': -0.73, 'LAW': -0.56, 'LOC': -0.77, 'MONEY': -0.31,
            'NORP': -0.7, 'ORDINAL': -0.66, 'ORG': -0.69, 'PERSON': -0.68,
            'PRODUCT': -1.0, 'QUANTITY': 1.0, 'TIME': -0.78, 'WORK_OF_ART': -0.95
        }

        self.whitelist = {
            'trump', 'elon musk', 'elon', 'musk', 'biden', 'putin', 'zelensky', 'xi', 'jinping',
            'macron', 'von der leyen', 'modi', 'meloni', 'lula da silva', 'netanyahu',
            'kamala harris', 'le pen', 'milei', 'scholz', 'twitter', 'x', 'tesla',
            'united nations', 'european union', 'nato', 'us'
        }

    def fit(self, X, y=None):
        self.nlp = spacy.load("en_core_web_sm")
        return self

    def _score_text(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return 0

        doc = self.nlp(text)
        score = 0

        for ent in doc.ents:
            label = ent.label_
            if label in self.entity_weights:
                weight = self.entity_weights[label]
                ent_text = ent.text.lower().strip()

                if self.use_whitelist:
                    if ent_text in self.whitelist:
                        weight += 0.5  # boost for whitelisted entities
                    else:
                        continue  # skip entities not in whitelist

                score += weight
        return score

    def transform(self, X):
        X_ = X.copy()
        text_columns = [col for col in X_.columns if col not in self.exclude_columns]
        score_df = pd.DataFrame(index=X_.index)

        for col in text_columns:
            new_col = f'{col}_ner_score'
            score_df[new_col] = X_[col].apply(self._score_text)

        return score_df


class NoveltyScoreExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_columns=None, window_size=30, max_features=1000, ngram_range=(1,2), max_df=0.97, norm='l2'):
        self.exclude_columns = exclude_columns or []
        self.window_size = window_size
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.norm = norm

    def fit(self, X, y=None):
        X_ = X.copy()
        text_columns = [col for col in X_.columns if col not in self.exclude_columns]
        X_['full_text'] = X_[text_columns].fillna('').agg(' '.join, axis=1)
        
        self.vectorizer = CountVectorizer(
            lowercase=True,
            max_features=self.max_features,
            stop_words='english',
            ngram_range=self.ngram_range,
            max_df=self.max_df
        )
        self.tfidf = TfidfTransformer(norm=self.norm)

        bow = self.vectorizer.fit_transform(X_['full_text'])
        self.tfidf.fit(bow)

        return self

    def transform(self, X):
        X_ = X.copy()
        text_columns = [col for col in X_.columns if col not in self.exclude_columns]
        X_['full_text'] = X_[text_columns].fillna('').agg(' '.join, axis=1)

        bow = self.vectorizer.transform(X_['full_text'])
        tfidf_matrix = self.tfidf.transform(bow).toarray()

        novelty_scores = []

        for i in range(tfidf_matrix.shape[0]):
            if i < self.window_size:
                novelty_scores.append(0)
            else:
                current_vector = tfidf_matrix[i].reshape(1, -1)
                window_vectors = tfidf_matrix[i - self.window_size:i]
                mean_vector = window_vectors.mean(axis=0).reshape(1, -1)

                similarity = cosine_similarity(current_vector, mean_vector)[0][0]
                novelty = 1 - similarity
                novelty_scores.append(novelty)

        # Rellenar primeros `window_size` días con ceros
        novelty_scores = [0] * self.window_size + novelty_scores[self.window_size:]
        novelty_scores = novelty_scores[:len(X_)]  # en caso de slicing por seguridad

        return pd.DataFrame({'novelty_score': novelty_scores}, index=X.index)


impact_score_pipeline = FeatureUnion([
    ('sentiment', SentimentScoreExtractor()),
    ('category', CategoryImpactExtractor()),
    ('ner', NERScoreExtractor()),
    ('novelty', NoveltyScoreExtractor())
])
