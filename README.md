✈️ Sentiment Analysis on Twitter Data – US Airlines

This project performs sentiment analysis on real-world Twitter data about US airlines. The model classifies tweets into positive, negative, or neutral categories using Natural Language Processing (NLP) techniques.

Problem Statement

Airlines often receive feedback from passengers via social media. Automatically detecting sentiment in tweets helps airlines track customer satisfaction and take action faster. This project uses machine learning models to predict the sentiment of tweets based on their content.

Technologies Used

Python
Pandas
NumPy
NLTK
Scikit-learn
TfidfVectorizer
RandomForestClassifier
MultinomialNB
Google Colab

Dataset

Source: Kaggle – Twitter US Airline Sentiment
URL: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
We use only the following columns:
airline_sentiment – the sentiment label (positive, negative, neutral)
text – the tweet content

Steps Performed

Loaded and reduced the dataset to only relevant columns

Preprocessed text by converting to lowercase, removing URLs, stopwords, and applying stemming

Extracted features using TF-IDF (maximum 3000 features)

Converted the processed text into numerical vectors

Split the data into training and testing sets (80 percent train, 20 percent test)

Trained two machine learning models:

Multinomial Naïve Bayes

Random Forest Classifier

Evaluated and compared their accuracy on the test set

Results

Both models were able to classify tweet sentiments with reasonable accuracy. The Random Forest model generally performed slightly better in accuracy, while Naïve Bayes trained faster.

How to Run

Download the dataset from the Kaggle link provided above

Upload the CSV file to Google Colab

Run the notebook sentiment_analysis_airlines.ipynb

Required libraries: nltk, pandas, sklearn

Evaluate the model performance using accuracy_score

Learning Outcome

Learned the core NLP preprocessing techniques using NLTK
Understood how to convert text into numerical vectors using TF-IDF
Practiced training and comparing classical machine learning models on text data
Improved ability to clean and extract features from real-world Twitter data

Credits

Dataset from Crowdflower on Kaggle
Project built as part of the SLIIT AI/ML Engineer Certification
