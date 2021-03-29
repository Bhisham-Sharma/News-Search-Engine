from flask import Flask, Response, render_template, request, jsonify
import pathlib
import pandas as pd
import functions as f
import csv
import pickle
import numpy as np
from fast_autocomplete import AutoComplete
import re
import json
import logging
from rank_bm25 import BM25Okapi

app = Flask(__name__)


@app.route('/')
def main_page():
    return render_template("index.html")

@app.route('/process', methods=["POST"])
def process():
    
    search_query = request.form.get("data")
    selected_category = request.form.get("category").lower()
    selected_sort = request.form.get("sort")
    
    df = pd.read_csv("./data/news_data.csv")

    doc_set = set()
    if selected_category == 'all':
        file = open("./data/index_dict.pkl", "rb")
        index_dict = pickle.load(file)
        doc_set = f.linearMergePosition(search_query, index_dict)
    else:
        file = open("./data/categorical_index_dict.pkl", "rb")
        category_index_dict = pickle.load(file)
        doc_set = f.linearMergePosition(search_query, category_index_dict[selected_category])


    result_df = df.loc[df['Doc_ID'].isin([doc_id for doc_id in doc_set])].copy()
    
    df_ = pd.read_csv("./data/full_data_and_cleaned_data.csv")

    tokenized_corpus = [df_['cleaned_data'][doc_id-1].split(" ") for doc_id in doc_set]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = f.preprocess_words(search_query).split(" ")
    doc_scores = bm25.get_scores(tokenized_query)

    result_df["Scores"] = doc_scores
    result_df.dropna(inplace=True)

    if selected_sort == "Relevance":
        result_df.sort_values('Scores', inplace=True, ascending=False)
    elif selected_sort == "Newest":
        result_df['Date'] =pd.to_datetime(result_df.date, yearfirst=True)
        result_df.sort_values('Date', inplace=True, ascending=False)
    else:
        result_df['Date'] =pd.to_datetime(result_df.date, yearfirst=True)
        result_df.sort_values('Date', inplace=True, ascending=True)

    all_links = []
    for headline, link, description, date in zip(result_df['headline'].values, result_df['link'].values, result_df['short_description'].values, result_df['date'].values):
        all_links.append([headline, link, str(description), date])
    return jsonify(all_links)
    
@app.route('/autocomplete', methods=["POST"])
def autocomplete():
    search = request.form.get("text")
    file = open("./data/suggestion_list.pkl", "rb")
    l = pickle.load(file)
    result = set()
    for element in l:
        for line in element:
            if line.startswith(search):
                result.add(line)
    return jsonify(list(result)[:10])
    


if __name__ == '__main__':

    # save json file to csv
    file = pathlib.Path("./data/news_data.csv")
    if file.exists() == False:
        df = pd.read_json("./data/News_Category_Dataset_v2.json",lines=True)
        df['Doc_ID'] = [x for x in range(1,df.shape[0]+1)]
        df.to_csv("./data/news_data.csv")

    # check for data exist or not
    file = pathlib.Path("./data/full_data_and_cleaned_data.csv")
    if file.exists() == False:
        with open("./data/full_data_and_cleaned_data.csv", mode='w', newline='', encoding="utf-8") as open_file:
            df = pd.read_csv("./data/News_data.csv")
            writer = csv.writer(open_file)
            writer.writerow(["full_data", "cleaned_data", "Doc_ID"])

            count = 1
            for title, description in zip(df['headline'], df['short_description']):
                d = str(title) + " " + str(description)
                writer.writerow([f.clean_data(d),f.preprocess_words(d),count])
                count += 1

    # check if index dict exist or not
    file = pathlib.Path("./data/index_dict.pkl")
    if file.exists() == False:
        df = pd.read_csv("./data/full_data_and_cleaned_data.csv")
        index_dict = {}

        for doc_id in df["Doc_ID"]:
            index = 0
            for word in str(df["cleaned_data"][doc_id-1]).split(" "):
                if word not in index_dict.keys():
                    index_dict[word] = [[doc_id,index]]
                else:
                    index_dict.get(word).append([doc_id,index])
                index += 1

        filename = "./data/index_dict.pkl"
        outfile = open(filename,'wb')
        pickle.dump(index_dict,outfile)
        outfile.close()

    # check if category wise index dict exist or not
    file = pathlib.Path("./data/categorical_index_dict.pkl")
    if file.exists() == False:
        df = pd.read_csv("./data/full_data_and_cleaned_data.csv")
        df_ = pd.read_csv("./data/news_data.csv")

        df_["cleaned_data"] = df["cleaned_data"]

        all_categories = np.unique(df_["category"])

        category_index_dict = {}

        for category in all_categories:
            df = df_.loc[df_['category'] == category].copy()
            temp_dict = {}
            for doc_id, cleaned_data in zip(df['Doc_ID'], df['cleaned_data']):
                index = 0
                for word in str(cleaned_data).split(" "):
                    if word not in temp_dict.keys():
                        temp_dict[word] = [[doc_id,index]]
                    else:
                        temp_dict.get(word).append([doc_id,index])
                index += 1
            category_index_dict[category.lower()] = temp_dict

        filename = "./data/categorical_index_dict.pkl"
        outfile = open(filename,'wb')
        pickle.dump(category_index_dict,outfile)
        outfile.close()

    # check if ranked phrases out of full corpus present or not
    file = pathlib.Path("./data/suggestion_list.pkl")
    if file.exists() == False:
        df = pd.read_csv("./data/full_data_and_cleaned_data.csv")
        all_phrases = []
        from rake_nltk import Rake
        rake = Rake()
        for data in df["full_data"]:
            rake.extract_keywords_from_text(data)
            all_phrases.append(rake.get_ranked_phrases())

        filename = "./data/suggestion_list.pkl"
        outfile = open(filename,'wb')
        pickle.dump(all_phrases,outfile)
        outfile.close()

    app.run(debug=True)