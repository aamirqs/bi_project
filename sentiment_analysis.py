import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect

sb.set()
nltk.download('wordnet')
nltk.download('omw-1.4')

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)

listings_df = pd.read_csv('/home/aamir/projects/bi_final_project/bi_project_dataset/listings.csv')
# calendar_df = pd.read_csv('/home/aamir/projects/bi_final_project/bi_project_dataset/calendar.csv')
reviews_df = pd.read_csv('/home/aamir/projects/bi_final_project/bi_project_dataset/reviews.csv')

reviews_df = reviews_df.dropna()
reviews_df['polarity_value'] = "Default"
reviews_df['neg'] = 0.0
reviews_df['pos'] = 0.0
reviews_df['neu'] = 0.0
reviews_df['compound'] = 0.0


def print_df_description(df):
    print("Columns:", df.columns.values)
    print("Head:")
    print(df.head(), "\n\n")
    print("Shape:", df.shape)


print_df_description(listings_df)

listings_df = listings_df[
    ['id', 'name', 'longitude', 'latitude', 'description', 'instant_bookable', 'neighborhood_overview',
     'neighbourhood_cleansed', 'host_id', 'host_name', 'host_since',
     'host_response_time', 'review_scores_rating', 'property_type', 'room_type', 'accommodates', 'bathrooms',
     'bedrooms', 'beds', 'reviews_per_month', 'amenities', 'number_of_reviews', 'price', 'host_is_superhost',
     'review_scores_value']]
print_df_description(listings_df)

# Replace NaN values with 0
listings_df.fillna(0, inplace=True)

# Create an empty prices list
prices = []
# Convert prices from listings_df into float values and append it in prices list
for i, p in listings_df['price'].items():
    p = p.replace(',', '')
    p = p.replace('$', '')
    p = float(p)
    prices.append(p)
listings_df['price'] = prices

listings_df = listings_df[listings_df.bedrooms > 0]
# listings_df = listings_df[listings_df.bathrooms > 0]
listings_df = listings_df[listings_df.accommodates > 0]
listings_df = listings_df[listings_df.price > 0]
listings_df = listings_df[listings_df.beds > 0]
listings_df = listings_df[listings_df.review_scores_rating > 0]
listings_df = listings_df[listings_df.reviews_per_month > 0]
listings_df = listings_df[listings_df.host_is_superhost == 't']
print_df_description(listings_df)

print("Number of room types :", len(listings_df["room_type"].unique()))
print("Number of property types :", len(listings_df["property_type"].unique()))
print()


####################Analysis

def clean_text(text):
    filtered_text = []
    punctuations = list(string.punctuation)
    ignore_char = ['\r', '\n', '', ' ', "'s"]
    lem = WordNetLemmatizer()
    text = text.lower()
    text = text.replace("/", ' ').replace(',', ' ').replace('?', ' ').replace('!', ' ').replace(':', ' ').strip()
    processed_token = [i for i in re.split('\s|\n|\t', text.strip()) if i != '']
    processed_token = list(filter(lambda s: any([c.isalnum() for c in s]), processed_token))
    processed_token = [i.replace('.', '').
                       replace(',', '').
                       replace('(', '').
                       replace(')', '').
                       replace('<', '').
                       replace('>', '').
                       replace(':', '').
                       replace('>', '').
                       replace('>', '').
                       replace('"', '').
                       replace('"', '').
                       replace("'", '').
                       replace("'", '').
                       replace("-", '').
                       replace("-", '') for i in processed_token]
    text = re.sub('<.*?>', '', text)
    text_tokens = word_tokenize(text)
    stop_words = stopwords.words('english')
    for t in text_tokens:
        if t not in stop_words and t not in punctuations and t not in ignore_char:
            filtered_text.append(lem.lemmatize(t))
    return '-'.join(filtered_text)


def create_word_cloud(cloud_words, filename=None):
    word_cloud = WordCloud(width=1000, height=700, background_color="white").generate(cloud_words)
    plt.figure(figsize=(18, 12))
    plt.imshow(word_cloud)
    plt.axis("off")
    if filename:
        plt.savefig(filename)
    plt.show()


def create_amenities_wordcloud():
    amenities_df = listings_df[['amenities', 'price', 'id', ]].sort_values('price', ascending=[0])
    all_words = ''

    for index, row in amenities_df.iterrows():
        amenity = row['amenities'].strip('][').replace('"', '').split(',')
        for a in amenity:
            all_words += "{} ".format(clean_text(a))
    create_word_cloud(all_words, 'amenities_wordcloud.png')


# Create a dataframe of the words that appear in the ammenities section of the most expensive listings


##### Description Wordcloud
def create_description_word_cloud():
    description_df = listings_df[['description', 'price']]
    description_df = description_df[pd.notnull(description_df['description'])]
    description_df = description_df[description_df['description'] != 0]
    top_df = description_df.sort_values('price', ascending=[0]).head(500)
    bot_df = description_df.sort_values('price', ascending=[1]).head(500)
    extended_word_stops = ['room', 'guest', 'thing', 'house', 'well', 'the', 'home', 'space', 'apartment', 'boston']

    top_desc = ' '.join(top_df['description']).lower()
    top_desc = re.sub('[0-9]', '', top_desc)
    top_desc = re.sub('<.*?>', '', top_desc)
    top_desc_cleaned = clean_text(top_desc)
    for e in extended_word_stops:
        top_desc_cleaned = top_desc_cleaned.replace(e, '')
    create_word_cloud(top_desc_cleaned, 'top_property_desc_word_cloud.png')

    bot_desc = ' '.join(bot_df['description']).lower()
    bot_desc = re.sub('[0-9]', '', bot_desc)
    bot_desc = re.sub('<.*?>', '', bot_desc)
    bot_desc_cleaned = clean_text(bot_desc)
    for e in extended_word_stops:
        bot_desc_cleaned = bot_desc_cleaned.replace(e, '')
    create_word_cloud(bot_desc_cleaned, 'btm_property_desc_word_cloud.png')


def create_review_cloud():
    print("Generating Word Cloud Based on Customer Reviews")
    '''
    listings_df['reviews'] = ''
    for i, l_row in listings_df.iterrows():
        review_list = []
        for j, r_row in reviews_df.iterrows():
            if r_row['listing_id'] == l_row['id']:
                review_list.append(r_row['comments'])
        listings_df.at[i, 'reviews'] = review_list
    '''
    out = listings_df.merge(reviews_df.groupby('listing_id').agg(list),
                            left_on='id', right_index=True, how='left')

    sr_df = out[pd.notnull(out['comments'])].sort_values('review_scores_value', ascending=[1])
    all_words = ''
    extended_stopwords = ["great", "one", "thank", "place", "stay", "airbnb"]

    for index, row in sr_df.iterrows():
        review_str = ' '.join(row['comments'])
        all_words += clean_text(review_str)
    for e in extended_stopwords:
        all_words = all_words.replace(e, '')
    print("Generating Word Cloud Based on Customer Reviews")
    create_word_cloud(all_words, 'reviews_wordcloud.png')


def run_sid_analyser(reviews_df, index, comment):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(comment)
    reviews_df.at[index, 'polarity_value'] = ss
    reviews_df.at[index, 'neg'] = ss['neg']
    reviews_df.at[index, 'pos'] = ss['pos']
    reviews_df.at[index, 'neu'] = ss['neu']
    reviews_df.at[index, 'compound'] = ss['compound']


def reviews_semantic_analysis():
    for i, row in reviews_df.iterrows():
        run_sid_analyser(reviews_df, i, row['comments'])
    reviews_df.to_csv('polarity_reviews.csv')


def detect_lang(sente):
    sente = str(sente)
    try:
        return detect(sente)
    except:
        return "None"


def polarity_ananlysis():
    for index, row in reviews_df.iterrows():
        lang = detect_lang(row['comments'])
        reviews_df.at[index, 'language'] = lang

    EngReviewsDF = reviews_df[reviews_df.language == 'en']

    EngReviewsDF.head(2)

    polarDF = EngReviewsDF[['pos']]
    polarDF = polarDF.groupby(pd.cut(polarDF["pos"], np.arange(0, 1.1, 0.1))).count()
    polarDF = polarDF.rename(columns={'pos': 'count_of_Comments'})
    polarDF = polarDF.reset_index()
    polarDF = polarDF.rename(columns={'pos': 'range_i'})
    for i, r in polarDF.iterrows():
        polarDF.at[i, 'RANGE'] = float(str(r['range_i'])[1:4].replace(',', ''))
        polarDF.at[i, 'Sentiment'] = 'positive'
    del polarDF['range_i']
    polarDF.head()

    polarDFneg = EngReviewsDF[['neg']]
    polarDFneg = polarDFneg.groupby(pd.cut(polarDFneg["neg"], np.arange(0, 1.1, 0.1))).count()
    polarDFneg = polarDFneg.rename(columns={'neg': 'count_of_Comments'})
    polarDFneg = polarDFneg.reset_index()
    polarDFneg = polarDFneg.rename(columns={'neg': 'range_i'})
    for i, r in polarDFneg.iterrows():
        polarDFneg.at[i, 'RANGE'] = float(str(r['range_i'])[1:4].replace(',', ''))
        polarDFneg.at[i, 'Sentiment'] = 'negative'
    del polarDFneg['range_i']
    for i, r in polarDFneg.iterrows():
        polarDF = polarDF.append(pd.Series([r[0], r[1], r[2]], index=['count_of_Comments', 'RANGE', 'Sentiment']),
                                 ignore_index=True)

    polarDFneg.head()

    polarDFnut = EngReviewsDF[['neu']]
    polarDFnut = polarDFnut.groupby(pd.cut(polarDFnut["neu"], np.arange(0, 1.0, 0.1))).count()
    polarDFnut = polarDFnut.rename(columns={'neu': 'count_of_Comments'})
    polarDFnut = polarDFnut.reset_index()
    polarDFnut = polarDFnut.rename(columns={'neu': 'range_i'})
    for i, r in polarDFnut.iterrows():
        polarDFnut.at[i, 'RANGE'] = float(str(r['range_i'])[1:4].replace(',', ''))
        polarDFnut.at[i, 'Sentiment'] = 'neutrl'
    del polarDFnut['range_i']

    for i, r in polarDFnut.iterrows():
        polarDF = polarDF.append(pd.Series([r[0], r[1], r[2]], index=['count_of_Comments', 'RANGE', 'Sentiment']),
                                 ignore_index=True)

    polarDFnut.head()

    plt.figure(figsize=(10, 10))
    sb.relplot(data=polarDF, x="RANGE", y="count_of_Comments", col="Sentiment", kind='line')
    plt.savefig('sentiment_analysis.png')


if __name__ == "__main__":
    # create_amenities_wordcloud()
    # create_description_word_cloud()
    create_review_cloud()
