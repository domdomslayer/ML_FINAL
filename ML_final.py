from bs4 import BeautifulSoup
import sklearn.feature_extraction.text as fe_text
import glob
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests

def bow_tfidf(docs):
    #TF-IDFで重みづけされたBag-of-Words形式の特徴ベクトルの生成
    vectorizer = fe_text.TfidfVectorizer(norm=None, stop_words='english')
    vectors = vectorizer.fit_transform(docs)
    return vectors.toarray(), vectorizer  

def pre_processing(text, mode):
    #分類クラスとなるジャンルの取得と、共通部分をカットする前処理
    if mode == "learning":
        genre_list.append(get_genre(text))
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
genre_list = [] #分類クラスであるジャンルのリスト

documents_list = glob.glob("./HTML/*") #ファイル「HTML」直下に全てのHTMLがある
for document in documents_list:
    fh = open(document, encoding='utf-8')
    fh_list.append(fh)
    #Beautiful Soupを用いてHTMLタグを削除
    pre_processing(BeautifulSoup(fh.read()).get_text(), "learning") 

#Bag-of-words形式の特徴ベクトルを生成
vectors_tfidf, vectorizer_tfidf = bow_tfidf(docs)

#String型の分類ラベルを数値に変換
le = LabelEncoder()
genre_id = le.fit_transform(genre_list)
print(le.classes_) #ラベルと数値の対応確認

np.insert(vectors_tfidf, vectors_tfidf.shape[1], genre_id, axis=1)
print(vectors_tfidf[1])

#データセットを訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(vectors_tfidf, genre_list)

#分類モデルには線形SVCを用いる
clf=svm.LinearSVC()
#clf=RandomForestClassifier()
clf.fit(X_train, y_train)
#print(clf.score(X_test, y_test))

for fh in fh_list:
    fh.close()