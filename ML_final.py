from bs4 import BeautifulSoup
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn.feature_extraction.text as fe_text
import glob
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import requests


def bow_tfidf(docs):
    #TF-IDFで重みづけされたBag-of-Words形式の特徴ベクトルの生成
    vectorizer = fe_text.TfidfVectorizer(norm=None, stop_words='english')
    vectors = vectorizer.fit_transform(docs)
    return vectors.toarray(), vectorizer  

def pre_processing(text, mode):
    #分類クラスとなるジャンルの取得と、共通部分をカットする前処理
    if mode == "learning":
        clfLabel_list.append(get_genre(text))
        docs.append(cut_common_part(text))
    elif mode == "predicting":
        return  cut_common_part(text), get_genre(text)

def get_genre(document):
    #HTMLファイルからジャンル情報を取得
    genre = ""
    word_list=document.split()
    for index, word in enumerate(word_list):
        if word == "Genre:":
            #第１ジャンルが「その他」の場合は第２ジャンルを取得
            if word_list[index + 1] == "Other,":
                genre = word_list[index + 2]
            else:
                genre = word_list[index + 1]
            if genre[-1] == ",":
                    genre = genre.strip(",")
            return genre

def cut_common_part(document):
    #全ページ共通な箇所をテキストから削除する
    text = ""
    isUnnecessary = True
    word_list = document.split()
    for index, word in enumerate(word_list):
        #「Read more」以下はいらない情報なのでカットする
        if word == "Read" and word_list[index + 1] == "more":
            break
        elif word == "points" and word_list[index - 4] == "Eligible":
            isUnnecessary = False
        elif isUnnecessary:
            continue
        #「Eligible for up to points」以前もいらない情報なのでカットする
        else:
            text += word + " "
    return text

#データセット収集効率化のためのジャンルの分布確認
def showClfLabelFreq(class_list):
    freq_list=[]
    for genre in class_list:
        freq_list.append(clfLabel_list.count(genre))
        #print(genre + " : " + str(clfLabel_list.count(genre)) +" data")
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_position([0.05,0.05,0.9,0.9])
    ax.tick_params(labelsize=8)
    ax.bar(range(len(freq_list)), freq_list,  tick_label=class_list)
    plt.show()


"""
引数に与えられたURLのゲームのジャンルを予測し、実際のジャンルと一緒に出力する関数
BoWで生成した特徴ベクトルの扱いが難しかったので没
"""
def predictOtherGame(url):
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    game_title = soup.find('title').text
    text, answer=pre_processing(soup.get_text(), "predicting")
    vectors, vectorizer = bow_tfidf(text)
    print("【" + game_title +"のゲームジャンル】")
    print("予測結果：" + clf.predict(vectors))
    print("実際：" + answer)


fh_list=[] #ファイルハンドラのリスト
docs=[] #ドキュメント本文（ゲームの紹介文）のリスト
clfLabel_list = [] #分類クラスであるジャンルのリスト
title_list = []

documents_list = sorted(glob.glob("./HTML/*")) #ファイル「HTML」直下に全てのHTMLがある
for document in documents_list:
    fh = open(document, encoding='utf-8')
    fh_list.append(fh)
    #Beautiful Soupを用いてHTMLタグを削除
    soup = BeautifulSoup(fh.read())
    pre_processing(soup.get_text(), "learning")
    title_list.append(document[7:-5]) #データフレームのindexに用いるためのタイトルリストも取得

#Bag-of-words形式の特徴ベクトルを生成
vectors_tfidf, vectorizer_tfidf = bow_tfidf(docs)

#String型の分類ラベルを数値に変換
le = LabelEncoder()
clfLabel_id = le.fit_transform(clfLabel_list)
#print(le.classes_) #ラベルと数値の対応確認

#データセットの分類クラス分布を確認する
#showClfLabelFreq(le.classes_)

#データセット分析用のデータフレームの生成
df_columns = vectorizer_tfidf.get_feature_names()
df_columns.append('"game_genre"')
df = pd.DataFrame(np.insert(vectors_tfidf, vectors_tfidf.shape[1], clfLabel_id, axis=1), index=title_list, columns=df_columns)
#print(df)

#データセットを訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(vectors_tfidf, clfLabel_list)

#分類モデルには線形SVCを用いる
clf=svm.LinearSVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

for fh in fh_list:
    fh.close()