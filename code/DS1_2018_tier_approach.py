import numpy as np
import pandas as pd
import csv


def get_predicted_bots(Final_result_Dict):
    """This function takes the graph in the form of dictionary and returns a list of predicted bot.

    args:
        Final_result_Dict: Graph of all the (screen_name_one,screen_name_two) in dictionary format
    return:
        final_bots: List of classified bots after applying threshold of 10 retweets
    """
    final_bots = set()
    for i in Final_result_Dict:
        if Final_result_Dict[i] > 10:
            final_bots.add(i[0])
            final_bots.add(i[1])

    final_bots = list(final_bots)
    return final_bots

def get_predicted_bots_tier_two(Final_result_Dict , bots_predicted):
    """This function takes the graph in the form of dictionary and returns a list of predicted bots.

        args:
            Final_result_Dict: Graph of all the (screen_name_one,screen_name_two) in dictionary format
        return:
            final_bots: List of classified bots after applying threshold of 10 retweets
    """
    final_bots = set(bots_predicted)
    for i in Final_result_Dict:
        if Final_result_Dict[i] > 10:
            final_bots.add(i[0])
            final_bots.add(i[1])

    return list(final_bots)



def save_bots_predicted(final_bots):
    with open("Bots_predicted_after_tier_two.csv", 'w', newline="") as f:
        thewriter = csv.writer(f)
        for i in final_bots:
            arr = [i]
            thewriter.writerow(arr)

def save_accurately_predicted(Accurately_predicted):
    # storing all accurately predicted bots after tier two
    with open("Bots_accurately_predicted_after_tier_two.csv", 'w', newline="") as f:
        thewriter = csv.writer(f)
        for i in Accurately_predicted:
            arr = [i]
            thewriter.writerow(arr)

def save_all_bots(bot_names):
    with open("all_bots.csv", 'w', newline="") as f:
        thewriter = csv.writer(f)
        for i in bot_names:
            arr = [i]
            thewriter.writerow(arr)


def build_graph(Data_for_second_threshold):

    """
    For predicting if there is some sort of CLBS present in the data frame we need to build a graph between all possible
    screen_names that are present in the dataset if they are suspect after applying first threshold

    The Task of building graph is divided in two parts
    1. Check which suspects retweeted the same tweet. ( Tid is unique for each and every tweet but, if two bots have
    retweeted a same tweet then the retweet_tid will be same for bot of them)
    2. Check if a suspect pair has shared the same retweet more than 10 times. ( If they have then there is a good chance
    that they are bots.)
    The Dataset is quite skewed so we may classify lot of actual human users as bots, but the overall aim is to accurately
    classify as many bots as we can.

    For step 1.
    we store all the unique names that have retweeted a particular tweet. we use retweet_id as key for the dictionary.
    type of Dict:
        key -> retweet_id
        value -> set of all the screen_name_from who retweeted that original tweet.

    For step 2:
    we make yet another dictionary and we save the suspect pairs that have retweeted the same tweet.
    we build a graph by iterating for each key in Dict and adding an edge between two screen_names that are in same set
    over here adding an edge is similar to adding one to the value in Final_result_Dict
    type of Dictionary:
        key -> (Screen_name_one, Screen_name_two)
        value -> number of times suspects, screen_name_one and screen_name_two shared the same tweet
    """
    #step one
    Dict = {}
    for i in Data_for_second_threshold:
        tid = i[0]
        retweet_tid = i[1]
        Screen_name_from = i[2]
        Screen_name_to = i[3]

        if retweet_tid not in Dict:
            Dict[retweet_tid] = set()
        Dict[retweet_tid].add(Screen_name_from)

    #step two
    Final_result_Dict = {}

    for i in Dict:
        Set_for_this_retweet = list(Dict[i])
        set_length = len(Set_for_this_retweet)

        if set_length > 1:
            for i in range(set_length - 1):
                for j in range(i + 1, set_length):
                    Screen_name_from_one = Set_for_this_retweet[i]
                    Screen_name_from_two = Set_for_this_retweet[j]
                    if (Screen_name_from_one, Screen_name_from_two) in Final_result_Dict:
                        Final_result_Dict[(Screen_name_from_one, Screen_name_from_two)] = Final_result_Dict[(
                            Screen_name_from_one, Screen_name_from_two)] + 1
                    elif (Screen_name_from_two, Screen_name_from_one) in Final_result_Dict:
                        Final_result_Dict[(Screen_name_from_two, Screen_name_from_one)] = Final_result_Dict[(
                            Screen_name_from_two, Screen_name_from_one)] + 1
                    else:
                        Final_result_Dict[(Screen_name_from_one, Screen_name_from_two)] = 1

    return Final_result_Dict


def get_all_botnames_and_data(year):
    """This function will return all the bots that are present in the dataset for a particular year
    args:
        year: The year for which we need the botnames from the shares_dataframe after applying threshold one.
    return:
        all_bots_that_year: A set of all the bot_names present in the dataset
        data_for_second_threshold: A list of all the
    """
    #in this step we read all the bots that could be present in the file from the botsname csv
    all_bots = pd.read_csv('botnames.csv')
    all_bot_names = set()
    for i in all_bots.itertuples():
        all_bot_names.add(i[2])

    #In this step we read the data for 2016/2018
    if year == 2018:
        Data = pd.read_csv('shares_df_2018.csv')
    elif year == 2016:
        Data = pd.read_csv('shares_df_2016.csv')

    all_bots_that_year = set()
    Data_for_second_threshold = []
    unique_accounts = set()

    for datarow in Data.itertuples():
        #check if the datarow is a suspect after first threshold or not
        #true value of threshold means the datarow is a suspect

        coordinated = datarow[6]
        Screen_name_from = datarow[3]
        unique_accounts.add(Screen_name_from)


        if Screen_name_from in all_bot_names:
            all_bots_that_year.add(Screen_name_from)

        if coordinated:
            tid = datarow[1]
            retweet_tid = datarow[2]
            Screen_name_to = datarow[4]
            Data_for_second_threshold.append([tid, retweet_tid, Screen_name_from, Screen_name_to])

    return all_bots_that_year , Data_for_second_threshold,len(unique_accounts)



def get_data_for_tier_two(bots_predicted , year):
    """This function returns the data for tier two calculation.

    args:
        bots_predicted: bots that are predicted by tier one
    return:
        Data_for_tier_two: list of data for tier two
    """

    final_bots = set(bots_predicted)
    Whole_data = pd.read_csv('SWM-dataset.csv')
    Data_for_tier_two = []

    for datarow in Whole_data.itertuples():
        time = str(datarow[5])
        Screen_name_to = datarow[4]
        if Screen_name_to in final_bots and time[:4] == str(year):
            tid = datarow[1]
            retweet_tid = datarow[2]
            Screen_name_from = datarow[3]
            Data_for_tier_two.append([tid, retweet_tid, Screen_name_from, Screen_name_to])


    return Data_for_tier_two

if __name__ == "__main__":
    bot_names , Data_for_second_threshold,unique_accounts = get_all_botnames_and_data(2018)

    Final_result_Dict_tier_one = build_graph(Data_for_second_threshold)

    #just applying threshold of 10 retweets between bots
    bots_predicted = get_predicted_bots(Final_result_Dict_tier_one)

    #get data for tier two
    Data_for_tier_two = get_data_for_tier_two(bots_predicted,2018)

    #graph for tier two

    Final_result_Dict_tier_two = build_graph(Data_for_tier_two)

    #bots_predicted_finally
    final_bots = get_predicted_bots_tier_two(Final_result_Dict_tier_two , bots_predicted)

    Accurately_predicted = []
    for i in final_bots:
        if i in bot_names:
            Accurately_predicted.append(i)


    len_Accurately_predicted = len(Accurately_predicted)
    len_bot_names = len(bot_names)
    len_final_bots = len(final_bots)

    #saving final bots predicted
    save_bots_predicted(final_bots)

    #saving accurately predicted
    save_accurately_predicted(Accurately_predicted)

    #saving all bots names in data
    save_all_bots(bot_names)

    #recall
    recall = len_Accurately_predicted/len_bot_names
    print(recall)

    #precision
    precision = len_Accurately_predicted/len_final_bots
    print(precision)

    #accuracy
    Total_predictions = unique_accounts
    TP = len_Accurately_predicted
    FN = len_bot_names - len_Accurately_predicted
    FP = len_final_bots - len_Accurately_predicted
    TN = Total_predictions - TP - FN - FP

    correct_predictions = TP + TN
    wrong_predictions = FN + FP
    Accuracy = correct_predictions / Total_predictions
    print(Accuracy)
