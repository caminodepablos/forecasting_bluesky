# Forecasting Bluesky

## Predicting user growth through activity and news impact analysis

### Description

This project seeks to analyze whether there is a significant correlation between political and social changes and the increase in user activity or migration to Bluesky.

### Context

Since the big change of Twitter (now X), many users have sought alternatives in other networks, mainly Threads (320M users), Mastodon (9M users) or Bluesky (34M users).
This scenario raises a crucial question: is Bluesky functioning as an escape valve against the polarization prevailing in the world?

### Hypothesis

There is a correlation between political and social changes and the increase of users on Bluesky. 

### Objectives

- Prediction of the increase of daily users in the social network.
- Prediction of the impact of political and social news on the growth of the social network.

## About this repository

```
forecasting_bluesky/
│
├── data_collection/
│   ├── blsk_users_activity_stats.csv
│   ├── news_data_classified_by_day.csv
│   ├── webscraping_bksy_feeds_news.ipynb
│   ├── bsky_feeds.csv
│   ├── bksy_news.csv
│   └── final_dataset_bsky_news.csv
│   
├── data_processing/
│   ├── data_preprocessing.ipynb
│   └── zero_shot_classification_news.ipynb
│   
├── data_analysis/
│   ├── EDA_bsky_stats.ipynb
│   ├── EDA_news_data.ipynb
│   └── EDA_full_dataset_bsky_news.ipynb
│  
├── data_modeling/
│   ├── bsky_linear_regression.ipynb
│   ├── bsky_logistic_regression.ipynb
│   └── impact_score/
│       ├── impact_score_components.ipynb
│       ├── impact_score_classification_model.ipynb
│       └── impact_score_xgbclass_model.pkl
│  
├── model_deployment/
│   └── forecasting-bluesky.py
│
├── forecasting_bluesky_code/
│   ├── __init__.py
│   ├── eda_plots.py
│   ├── impact_score.py
│   ├── metrics_plots.py
│   └── preprocessing.py
```

## References
- [See bibliography & references](https://github.com/caminodepablos/forecasting_bluesky/blob/main/references.md)

## Contributing

Feel free to open issues or contribute improvements via pull requests!

📧 Contact: For any questions, reach out via GitHub Issues or email me at caminodepablos@gmail.com

Thanks!
