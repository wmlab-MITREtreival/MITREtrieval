# source: https://github.com/ksatvat/EXTRACTOR
import re
import spacy
import numpy as np
import pandas as pd
#from tika import parser
from nltk import sent_tokenize
nlp = spacy.load("en_core_web_lg")
from tqdm import tqdm
from Preprocess.Ontology_Feature.Actor_Info_Extract import group_software_parser
from Preprocess.Ontology_Feature.Threat_Action_Extract import V_O_parser
from Preprocess.Ontology_Feature.Dict import load_lists,fpath
import pickle
from Preprocess.Ontology_Feature.Passive2Active import pass2acti
Ontology_Feature_Dict={} ## Dictionary for storing feature of ontology


from tensorflow.python.client import device_lib
print("***GPU*** ",device_lib.list_local_devices())



'''
 * delete_brackets()-Delete brackets in CTI report
 * @text: Text in CTI report
'''
def delete_brackets(text):
	text = text.replace("[","")
	text = text.replace("]", "")
	text = text.replace("<", "")
	text = text.replace(">", "")
	text = text.replace("(", "")
	text = text.replace(")", "")
	return text

def all_sentences(string):
	nltk_sentences = sent_tokenize(string)

	all_sentences_list = []
	for i in nltk_sentences:
		i.rstrip()
		if i.endswith(".") and "\n" not in i:
			all_sentences_list.append(i)
		elif "\n" in i:
			i.split("\n")
			for j in i.split("\n"):
				all_sentences_list.append(j)
	
	return all_sentences_list

def remove_analysis_by(txt):
    var = "Analysis by"
    lst = all_sentences(txt)
    for i in lst:
        if i.startswith(var):
            lst.remove(i)
    return lst

def perform_following_action(txt):  # When Virus:Win32/Funlove.4099 runs, it performs the following actions:
    perform_following_action_list = load_lists(fpath)['MS_PFA']
    perform_following_action_list = perform_following_action_list.replace("'", "").strip('][').split(', ')
    lst = remove_analysis_by(txt)
    for i in lst:
        for j in perform_following_action_list:
            if j in i:
                lst.remove(i)
                break
    return lst
def on_the_windows_x_only(txt):
	# on_the_windows_x_list = load_lists_microsoft.on_the_windows_x_lst()
	on_the_windows_x_list = load_lists(fpath)['MS_OTW']
	on_the_windows_x_list = on_the_windows_x_list.replace("'", "").strip('][').split(', ')
	lst = perform_following_action(txt)

	for i in lst:
		for j in on_the_windows_x_list:
			if j == i:
				lst.remove(i)
				#break
	return lst
def removable_token(txt):  # When Virus:Win32/Funlove.4099 runs, it performs the following actions:
	removable_token_list = load_lists(fpath)['RTL']
	removable_token_list = removable_token_list.replace("'", "").strip('][').split(', ')
	lst = on_the_windows_x_only(txt)

	for id, value in enumerate(lst):
		for j in removable_token_list:
			if value.strip().startswith(j):  #### definetly remember we should use only startswith()for proper matching
				# lst.remove(value)
				lst[id] = value.replace(j, " ")
				# break
	return lst

def handle_title(mylist_):  # handles titles and "." of the previous sentence
	titles_list = load_lists(fpath)['MS_TITLES']
	titles_list = titles_list.replace("'", "").strip('][').split(', ')
	lst_handled_titles = []
	lst = list(filter(lambda a: a != "", mylist_))[::-1]
	lst = list(filter(lambda a: a != " ", lst))
	lst = list(filter(lambda a: a != "", lst))
	for indx, val in enumerate(lst):
		lst[indx] = val.strip()
		if val=='':
			del lst[indx]
	l = len(lst)
	for index, item in enumerate(lst):
		if index < l - 1:
			if item in titles_list:
				x = lst[index + 1]
				if lst[index + 1] not in titles_list:
					if len(lst[index + 1].rstrip()) >=1:  # inja
						if lst[index + 1].rstrip()[-1] != ".":
							if lst_handled_titles:
								if lst[index + 1] + "." != lst_handled_titles[-1]:
									lst_handled_titles.append(lst[index + 1] + ".")
							else:
								lst_handled_titles.append(lst[index + 1] + ".")
						else:
							if lst_handled_titles:
								if lst[index + 1] != lst_handled_titles[-1]: # mahshid added n
									lst_handled_titles.append(lst[index + 1])
							else:
								lst_handled_titles.append(lst[index + 1])
				else:
					pass
			else:
				if lst_handled_titles:
					if item + "." not in lst_handled_titles and item !=  lst_handled_titles[-1]:
						lst_handled_titles.append(item)
				else:

					lst_handled_titles.append(item)
		else:
			if item not in titles_list:
				if item != lst_handled_titles[-1]:
					lst_handled_titles.append(item)
	lst = lst_handled_titles[::-1]
	lst = list(filter(lambda a: a != " ", lst))
	return list(filter(lambda a: a != "", lst))
def zero_word_verb(string):
	main_verbs = load_lists(fpath)['verbs']
	main_verbs = main_verbs.replace("'", "").strip('][').split(', ')
	doc = nlp(string.strip())
	if not (doc[0].tag_ == "MD") and\
			not (doc[0].tag_ == "VB" and
					str(doc[0]).lower() in main_verbs) and\
			not (doc[0].tag_ == "VB" and
					str(doc[0]).lower() not in main_verbs) and\
			not(str(doc[0]).lower() in main_verbs):
		return False
	else:
		return  True

def iscaptalized(sentence):
    if sentence.strip()[0].isupper() == True:
        return True
    else:
        return False
def sentence_characteristic(sentence):
    doc = nlp(sentence)
    if len(sentence.split(" ")) > 3:
        count_verb, count_noun = 0, 0
        for token in doc:
            if token.pos_ == "VERB":
                count_verb += 1
            if token.pos_ == "NOUN":
                count_noun += 1
        if count_verb >= 1 and count_noun >= 2:
            return True
    else:
        return False
def likely_sentence_characteristic(sentence):
    doc = nlp(sentence)
    if zero_word_verb(sentence) == True:
        if len(sentence.split(" ")) > 3:
            return True

        return "UNKNOWN"

    if iscaptalized(sentence) == True:
        if len(sentence.split(" ")) > 3:
            count_verb, count_noun = 0, 0
            for token in doc:
                if token.pos_ == "VERB":
                    count_verb += 1
                if token.pos_ == "NOUN":
                    count_noun += 1
            if count_verb >= 1 and count_noun >= 2:
                return True
    else:
        return False


def remove_non_ascii(s):
    return "".join(c for c in s if ord(c)<128)

'''
 * query_ontology()-Utilize features to query ontology
 * @text: Text in CTI report
'''
def query_ontology(text):
	group_list=[]
	software_list=[]
	text=remove_non_ascii(text)
	text=delete_brackets(text).strip(" ")
	sentence_list=sent_tokenize(text)
	all_node,v_o=V_O_parser(sentence_list)
	group,software=group_software_parser(text)
	return group,software,all_node,v_o

'''
 * Feature_extract_ontology()-Extract query node of every document
 * @data_df: Dataframe consists of CTI reports
 * @threshold: Technique threshold
 * @train: Whether training or testing
 * @return: Dataframe consist of group, software, verb object of each CTI reports
'''
def Feature_extract_ontology(data_df,threshold:str,train=''): ##this is for extracting feature of ontology
	g_s_df=pd.DataFrame(data_df, columns = ['Text','list'])
	g_s_df['list']=list(g_s_df['list'])
	all_group_list=[]
	all_software_list=[]
	all_vo_pair=[]
	all_sentence_extract=[]
	data_df=data_df['Text'] 
	print(train)
	for text in tqdm(data_df):
		group,software,all_node,v_o=query_ontology(text)
		all_group_list.append(group)
		all_software_list.append(software)
		all_vo_pair.append(v_o)
		all_sentence_extract.append(all_node)
		del group,software, v_o,all_node
	g_s_df['v_o']=all_vo_pair
	g_s_df['srl']=all_sentence_extract
	g_s_df['Group']=all_group_list
	g_s_df['Soft']=all_software_list
	f=open('ontology_feature_'+threshold+train,'wb')
	pickle.dump(g_s_df,f)
	f.close()
	return g_s_df

'''
 * query_node_extract()-Extract query node
 * @doc: Text in CTI report
'''
def query_node_extract(doc):
	group,software,all_node,v_o=query_ontology(doc)
	'''
	group: Group match 
	software: Software match
	all_node: SRL result
	v_o: Verb object pair
	'''
	return group,software,all_node,v_o