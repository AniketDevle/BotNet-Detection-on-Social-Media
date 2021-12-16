import numpy as np
import pandas as pd
import csv
from collections import Counter  
from urllib.parse import urlparse

def get_bots_from_Ds2():
    Data = pd.read_csv('shares_df_dat2.csv')
    Data = Data.drop_duplicates()
    all_bots_in_2018_dataset = set()
    unique_accounts= set()

    Data_for_second_threshold = []
    for datarow in Data.itertuples():
        coordinated = datarow[8]
        Screen_name_from = datarow[6]
        unique_accounts.add(Screen_name_from)

        if coordinated:
            tid = datarow[1]
            retweet_tid = datarow[4]
            Screen_name_to = datarow[5]

#     print(tid,retweet_tid,Screen_name_from,Screen_name_to)
    Data_for_second_threshold.append([tid, retweet_tid, Screen_name_from, Screen_name_to])
    Dict = {}
    for i in Data_for_second_threshold:

        retweet_tid = i[1]
        Screen_name_from = i[2]

        if retweet_tid not in Dict:
            Dict[retweet_tid] = set()
        Dict[retweet_tid].add(Screen_name_from)

    Final_result_Dict = {}

    for ind,i in enumerate(Dict):
        Set_for_this_retweet =list(Dict[i])
        set_length = len(Set_for_this_retweet)

        if set_length > 1:
            for i in range(set_length - 1):
                for j in range(i + 1, set_length):
                    Screen_name_from_one = Set_for_this_retweet[i]
                    Screen_name_from_two = Set_for_this_retweet[j]
                    list_names = [Screen_name_from_one,Screen_name_from_two]
                    list_names.sort()
                    if tuple(list_names) in Final_result_Dict:
                        Final_result_Dict[tuple(list_names)] = Final_result_Dict[tuple(list_names)] + 1
                    
                    else:
                        Final_result_Dict[tuple(list_names)] = 1


    final_bots = set()
    for i in Final_result_Dict:
        if Final_result_Dict[i] > 10:
            final_bots.add(i[0])
            final_bots.add(i[1])

    final_bots = list(final_bots)
    with open("Ds-2_predicted_bots.txt" , 'w' ) as filehandle:
        for listitem in final_bots:
            filehandle.write('%s\n' % listitem)

def get_most_boosted_domains(Data):
    being_boosted = {}
    for datarow in Data.itertuples():
        coordinated = datarow[8]
        Screen_name_from = datarow[6]
        try :
            url =  urlparse(datarow[7]).netloc
            if coordinated and Screen_name_from in final_bots:
                screen_name_to = datarow[5]
                if url not in being_boosted:
                    being_boosted[url] = 0
                being_boosted[url] = being_boosted[url] + 1
        except:
            continue


    names_boosted = list(being_boosted.keys())
    most_boosted = (sorted(being_boosted.items(), key=lambda item: item[1],reverse=True))
    return most_boosted[:10]

def get_most_boosted_accounts(Data):
    being_boosted = {}
    for datarow in Data.itertuples():
        coordinated = datarow[8]
        Screen_name_from = datarow[6]
        if coordinated and Screen_name_from in final_bots:
            screen_name_to = datarow[5]
            if screen_name_to not in being_boosted:
                being_boosted[screen_name_to] = 0
            being_boosted[screen_name_to] = being_boosted[screen_name_to] + 1

    names_boosted = list(being_boosted.keys())
    most_boosted = (sorted(being_boosted.items(), key=lambda item: item[1],reverse=True))
    return most_boosted[:10]




