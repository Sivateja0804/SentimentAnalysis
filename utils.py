
import re
import numpy as np
import pandas as  pd

import gensim
import gensim.corpora as corpora
import seaborn as sns
from gensim.utils import simple_preprocess
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score,plot_confusion_matrix
from gensim.models import CoherenceModel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
import seaborn as sb
from sklearn.manifold import TSNE
import mlxtend
from mlxtend.evaluate import bias_variance_decomp
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
nltk.download('averaged_perceptron_tagger')
import contractions
lemmatizer = WordNetLemmatizer()

def get_data_from_csv(trainPath):
    train_df = pd.read_csv(trainPath)
    return train_df


def get_world_cloud(data_textOnly):
    wordcloud = WordCloud().generate(data_textOnly)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(data_textOnly)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def preprocess_text(sen):
    snow=nltk.stem.SnowballStemmer('english')
    sen=sen.lower()
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    words=[snow.stem(word) for word in sentence.split()]
    sentence=" ".join(words)
    return sentence.strip()

def filter_data(df):
    filter = df["Content"] != ""
    df = df[filter]
    df = df.dropna()
    return df

def format_data(df,pred="single"):
    X = []
    sentences = list(df["Content"])
    for sen in sentences:
        X.append(preprocess_text(sen))
    y = df["Label"].values
    if pred=="single":
        T=[]
        for val in y:
            if val=="TRUE" or val=="mostly-true":
                T.append(1)
            else:
                T.append(0)
        y=np.array(T)
    return X,y

def cleanHTMLTags(sentence):
    return re.sub(re.compile('<.*?>'),' ',sentence)
def cleanPunctuations(sentence):
    return re.sub(r'[.|,|)|(|\|/]',r' ',re.sub(r'[?|!|\'|"|#]',r' ',sentence))

def data_preprocess(trainPath):
    data=get_data_from_csv(trainPath)
    rating3_data=data[data['Score']==3]
    print("resulting filtered data length after dropping rating 3 should be "+str(len(data)-len(rating3_data)))
    filtered_data=data[data['Score']!=3]
    print("Filtered data length is "+str(len(filtered_data)))
    #lets assign proper score 1- for 4 and 5 ratings and 0- for 1 and 2 ratings
    def assign_score(Score):
        return 0 if Score<3 else 1
    filtered_data['sentiment']=filtered_data['Score'].apply(assign_score)
    filtered_data.drop(['Score'],axis=1,inplace=True)
    filtered_data=filtered_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"},keep='first',inplace=False)
    filtered_data=filtered_data[filtered_data.HelpfulnessNumerator<=filtered_data.HelpfulnessDenominator]
    stop=set(stopwords.words('english'))
    snow=nltk.stem.SnowballStemmer('english')
    str1=" "
    final_string=[]
    for sentence in filtered_data["Text"].values:
        filtered_sentence=[]
        sentence=cleanHTMLTags(sentence)
        sentence=contractions.fix(sentence)
        sentence=cleanPunctuations(sentence)
        words=[snow.stem(word) for word in sentence.split() if word not in stop]
        str1=" ".join(words)
        final_string.append(str1)
    filtered_data["Text"]=final_string
    filtered_data=filtered_data.dropna()
    review=filtered_data["Text"]
    sentiment=filtered_data["sentiment"]
    return review,sentiment,filtered_data
    
        


def get_accuracy(review,sentiment,model,lr,show_coefficients=False):
    review_model=model.fit_transform(review)
    review_train,review_test,target_train,target_test=train_test_split(review_model,sentiment,test_size=0.2,random_state=0)
    final_model=lr.fit(review_train,target_train)
    yhat=final_model.predict(review_test)
    print("Accuracy :", np.mean(yhat == target_test))
    print(classification_report(target_test, yhat))
    confusionMatrix(target_test,yhat)
    if show_coefficients:
        df=pd.DataFrame({'Word':model.get_feature_names(),'Coefficient':final_model.coef_.tolist()[0]}).sort_values(['Coefficient','Word'],ascending=[0,1])
        print("-----------------Top 20 positive words------------")
        print(df.head(20).to_string(index=False))
        print("-----------------Top 20 negative words------------")
        print(df.tail(20).to_string(index=False))


def confusionMatrix(testset,predicted):
    #Confusion matrix
    print("======Confusion Matrix======")
    matrix = confusion_matrix(testset,predicted)
    print('\n',matrix)

    pd.crosstab(np.array(predicted), np.array(testset), rownames=['Actual'], colnames=['Predicted'], margins=True)

    p = sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
        
def plot_data_using_TSNE(X,y,vec="count"):
    # Initializing vectorizer for bigram
    if vec=="count":
        count_vect = CountVectorizer(ngram_range=(1,2),max_features=300)
    else:
        count_vect = TfidfVectorizer(ngram_range=(1,2),max_features=300)

    # Initializing standard scaler
    std_scaler = StandardScaler(with_mean=False)

    # Creating count vectors and converting into dense representation
    sample_points = X
    sample_points = count_vect.fit_transform(sample_points)
    sample_points = std_scaler.fit_transform(sample_points)
    sample_points = sample_points.todense()

    # Storing class label in variable
    labels = y

    # Getting shape
    print(sample_points.shape, labels.shape)
    
    tsne_data = sample_points
    tsne_labels = labels

    # Initializing with most explained variance
    model = TSNE(n_components=2, random_state=15, perplexity=50, n_iter=2000)

    # Fitting model
    tsne_data = model.fit_transform(tsne_data)

    # Adding labels to the data point
    tsne_data = np.vstack((tsne_data.T, tsne_labels)).T

    # Creating data frame
    tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))

    # Plotting graph for class labels
    sb.FacetGrid(tsne_df, hue='label', size=5).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
    plt.title("TSNE with default parameters")
    plt.xlabel("Dim_1")
    plt.ylabel("Dim_2")
    plt.show()
 
def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
 

 
 
def swn_polarity(text):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """
 
    sentiment = 0.0
    tokens_count = 0
 
 
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
 
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
 
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
 
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
 
            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
 
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1
 
    # judgment call ? Default to positive or negative
    if not tokens_count:
        return 0
 
    # sum greater than 0 => positive sentiment
    if sentiment >= 0:
        return 1
 
    # negative sentiment
    return 0



def append_senti_to_sent(train):
    
    def senti_append(sentences): 
        X = []  
        for i in range(len(sentences)):
            sen=sentences[i]
            if swn_polarity(sen)==1:
                X.append(1)
            else:
                X.append(0)
        return X
    X=senti_append(train)
    return X


def append_senti_to_vect(train,test):
    
    def senti_append(sentences): 
        senti = []  
        for i in range(len(sentences)):
            sen=sentences[i]
            if swn_polarity(sen)==1:
                senti.append(1)
            else:
                senti.append(0) 
        return senti
    train_senti=senti_append(train)
    test_sent=senti_append(test)
    return train_senti,test_sent

