import tweepy
import pandas as pd
import json
import yaml
import csv


def tweepy_authentication():
    '''
    This function uses consumer key & secret and access tokens
    to authenticate and access Twitter API.

    Arguments: None
    Returns: API

    Example: 
    >>> api = tweepy_authentication()
    '''

    # credentials are stored in file 'credentials.yml':
    credentials = yaml.load(open('credentials.yml'))
    consumer_key = credentials['API_credentials']['consumer_key']
    consumer_secret = credentials['API_credentials']['consumer_secret']
    access_token = credentials['API_credentials']['access_token']
    access_secret = credentials['API_credentials']['access_secret']

    twitter_auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    twitter_auth.set_access_token(access_token, access_secret)
    twitter_api = tweepy.API(twitter_auth, wait_on_rate_limit=False, wait_on_rate_limit_notify=True)

    return twitter_api


def tweepy_save_Allresults_from_id(twitter_api, no_response_to_retrieve):
    '''
    This function returns twitter_api responses for ids
    in the given dataset 'SWM-dataset.csv'. 
    To specify the no of ids you would like to get results for, 
    use 'no_response_to_retrieve' parameter.

    Arguments: 
        twitter_api: Twitter API
        no_response_to_retrieve: number of tweet ids in SWM-dataset.csv that you'd
                                 like to get results for.
    Returns: None

    Example: 
    >>> no_response_to_retrieve = 100
    >>> tweepy_save_Allresults_from_id(twitter_api, no_response_to_retrieve)
    '''

    # input file which has ids specified in it (given to us by professor):
    datarows = pd.read_csv('dataset/SWM-dataset.csv', nrows=no_response_to_retrieve)

    json_api_response = {'results':[]}
    for datarow in datarows.itertuples():
        try:
            tweet = twitter_api.get_status(datarow[1])            
            json_api_response['results'].append(tweet._json)
        except tweepy.error.TweepError:
            print('protected tweet, skipping')

    # dumping the api response to output file "tweepy_reponses_dataset.json":
    with open('dataset/tweepy_reponses_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(json_api_response, f, ensure_ascii=False, indent=4)


def tweepy_save_text_from_id():
    '''
    This function uses the results output file created in 
    tweepy_save_Allresults_from_id() function to retrieve
    text and id of reponses and saves it to output file:
    'text_id_dataset.csv'

    Arguments: None
    Returns: None

    Example: 
    >>> tweepy_save_text_from_id()
    '''
    try:
        # input file created in tweepy_save_Allresults_from_id()
        # containing full result for a tweet:
        f = open('dataset/tweepy_reponses_dataset.json')
        json_api_response = json.load(f)

        # output file 'text_id_dataset.csv' with texts and tweet ID:
        with open('dataset/text_id_dataset.csv','a', encoding='utf-8') as opfile:
            writer = csv.writer(opfile)
            for tweet in json_api_response['results']:
                try:
                    writer.writerow((tweet['id'],tweet['text']))
                except:
                    pass
    except:
        print ('Please make sure you have tweepy_reponses_dataset.json in your dataset folder')


if __name__ == '__main__':

    twitter_api = tweepy_authentication()
    no_response_to_retrieve = 1000
    tweepy_save_Allresults_from_id(twitter_api,no_response_to_retrieve)
    tweepy_save_text_from_id()
    
