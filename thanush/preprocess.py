import json 
with open ('Medical_dataset/dieseas.json','r')as f:
    intents = json.load(f)


import pandas as pd
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
#ibstaling english language model
nlp = spacy.load('en_core_web_sm')

def preprocess(doc):
    doc=doc.replace("'t",' not')
    nlp_doc=nlp(doc)
    d=[]
    for token in nlp_doc:
        if(not token.text.lower()  in STOP_WORDS and  token.text.isalpha()):
            d.append(token.lemma_.lower() )
    return ' '.join(d)
#english stop words enerting
stp=stopwords.words('english')
stp.remove('not')

def preprocess_sent(sent):
    sent=sent.replace("'t",' not')
    t=nltk.word_tokenize(sent)
    return ' '.join([lemmatizer.lemmatize(w.lower()) for w in t if (w not in stp and w.isalpha())])
preprocess_sent("i can't breath")
sent=[]
app_tag=[]
for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        sent.append(preprocess_sent(pattern))
        app_tag.append(tag)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(sent)
feature_names = vectorizer.get_feature_names_out()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
vocab=list(df.columns)
#extra print to evaluate
#print (df.head)
def bag_of_words(tokenized_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
xy_test = [
    (['can',"'t", 'think', 'straight'], 'altered_sensorium'),
    (['suffer', 'from', 'anxeity'], 'anxiety'),
    (['suffer', 'from', 'anxeity'], 'anxiety'),
    (['bloody', 'poop'], 'bloody_stool'),
    (['blurred', 'vision'], 'blurred_and_distorted_vision'),
    (['can', "'t", 'breathe'], 'breathlessness'),
    (['Yellow', 'liquid', 'pimple'], 'yellow_crust_ooze'),
    (['lost', 'weight'], 'weight_loss'),
    (['side', 'weaker'], 'weakness_of_one_body_side'),
    (['watering', 'eyes'], 'watering_from_eyes'),
    (['brief', 'blindness'], 'visual_disturbances'),
    (['throat', 'hurts'], 'throat_irritation'),
    (['extremities', 'swelling'], 'swollen_extremeties'),
    (['swollen', 'lymph', 'nodes'], 'swelled_lymph_nodes'),
    (['dark', 'under', 'eyes'], 'sunken_eyes'),
    (['stomach', 'blood'], 'stomach_bleeding'),
    (['blood', 'urine'], 'spotting_urination'),
    (['sinuses', 'hurt'], 'sinus_pressure'),
    (['watery', 'from', 'nose'], 'runny_nose'),
     (['have', 'to', 'move'], 'restlessness'),
    (['red', 'patches', 'body'], 'red_spots_over_body'),
    (['sneeze'], 'continuous_sneezing'),
    (['coughing'], 'cough'),
    (['skin', 'patches'], 'dischromic_patches'),
    (['skin', 'bruised'], 'bruising'),
    (['burning', 'pee'], 'burning_micturition'),
    (['hurts', 'pee'], 'burning_micturition'),
    (['Burning', 'sensation'], 'burning_micturition'),
    (['chest', 'pressure'], 'chest_pain'),
    (['pain', 'butt'], 'pain_in_anal_region'),
    (['heart', 'bad', 'beat'], 'palpitations'),
    (['fart', 'lot'], 'passage_of_gases'),
    (['cough', 'phlegm'], 'phlegm'),
    (['lot', 'urine'], 'polyuria'),
    (['Veins', 'bigger'], 'prominent_veins_on_calf'),
    (['Veins', 'emphasized'], 'prominent_veins_on_calf'),
    (['yellow', 'pimples'], 'pus_filled_pimples'),
    (['red', 'nose'], 'red_sore_around_nose'),
     (['skin', 'yellow'], 'yellowish_skin'),
    (['eyes', 'yellow'], 'yellowing_of_eyes'),
    (['large', 'thyroid'], 'enlarged_thyroid'),
    (['really', 'hunger'], 'excessive_hunger'),
    (['always', 'hungry'], 'excessive_hunger'),
]
def preprocess_test(sent):
    return [lemmatizer.lemmatize(w.lower()) for w in sent if (w not in set(stopwords.words('english')) and w.isalpha())]
preprocess_sent(' '.join(xy_test[0][0]))
y_true=[]
y_pred=[]
for x,y in xy_test:
    y_true.append(y)
    p=preprocess_sent(' '.join(x))
    print(p)
    bow=np.array(bag_of_words(p,vocab))
    #    bow=vectorizer.transform(p).toarray()
    res=cosine_similarity(bow.reshape((1, -1)), df).reshape(-1)
    y_pred.append(app_tag[np.argmax(res)])
    y_pred
    error=0
for i in range(len(y_pred)):
    if y_pred[i]!=y_true[i]:
        error+=1
        1-error/len(y_true)
        x=['breathe']
p=preprocess_sent(' '.join(x))
bow=np.array(bag_of_words(p,vocab))
res=cosine_similarity(bow.reshape((1, -1)), df).reshape(-1)
app_tag[np.argmax(res)]
a=np.argsort(res)[::-1][:2].tolist()
#deployment
df=pd.read_csv('tfidfsymptoms.csv')
vocab=list(df.columns)
#print(df.columns)
import joblib
#getting an error in loading
knn= joblib.load('knn.pkl')  
#knn_from_joblib.predict(X_test) 
def bag_of_words(tokenized_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
def predictSym(sym,vocab,app_tag):
    sym=preprocess_sent(sym)
    bow=np.array(bag_of_words(sym,vocab))
    res=cosine_similarity(bow.reshape((1, -1)), df).reshape(-1)
    order=np.argsort(res)[::-1].tolist()
    possym=[]
    for i in order:
        if i < len(app_tag) and app_tag[i].replace('_',' ') in sym:
            return app_tag[i],1
        if i < len(app_tag) and app_tag[i] not in possym and res[i]!=0:
            possym.append(app_tag[i])
    return possym,0

    predictSym('i have skin erumptions',vocab,app_tag)
#importing trained csv
df_tr=pd.read_csv('Medical_dataset/Training.csv')
#print(df_tr.head)
#extra added defining function
def clean_symp(sym):
    intents=df_tr.iloc[:,-1].to_list()
all_symp_col=list(df_tr.columns[:-1])
all_symp=[clean_symp(sym) for sym in (all_symp_col)]
#recoit client_symptoms et renvoit un dataframe avec 1 pour les symptoms associees
def OHV(cl_sym,all_sym):
    l=np.zeros([1,len(all_sym)])
    for sym in cl_sym:
        l[0,all_sym.index(sym)]=1
    return pd.DataFrame(l, columns =all_symp)

def contains(small, big):
    a=True
    for i in small:
        if i not in big:
            a=False
    return a

def possible_intents(l,intents):
    poss_dis=[]
    for dis in set(intents):
        if contains(l,symVONintents(df_tr,dis)):
            poss_dis.append(dis)
    return poss_dis
#recoit une maladie renvoit tous les sympts
def symVONintents(df,intents):
    ddf=df[df.prognosis==intents]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()
    
def clean_symp(sym):
    return sym.replace('_',' ').replace('.1','').replace('(typhos)','').replace('yellowish','yellow').replace('yellowing','yellow')
symVONintents(df_tr,'Allergy')
def getInfo():
    # name=input("Name:")
    print("Your Name \n\t\t\t\t\t\t",end="=>")
    name=input("")
    print("hello ",name)
    return str(name)
import csv

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

def getDescription():
    global description_list
    with open('Medical_dataset/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)




def getSeverityDict():
    global severityDictionary
    with open('Medical_dataset/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('Medical_dataset/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp))>13):
        return 1
        print("You should take the consultation from doctor. ")
    else:
        return 0
        print("It might not be that bad but you should take precautions.")
getSeverityDict()
getprecautionDict()
getDescription()
def main_sp(name):
    #main Idea: At least two initial sympts to start with
    
    #get the 1st syp ->> process it ->> check_pattern ->>> get the appropriate one (if check_pattern==1 == similar syntaxic symp found)
    print("Hi Mr/Ms "+name+", can you describe you main symptom ?  \n\t\t\t\t\t\t",end="=>")
    sym1 = input("")
    psym1,find=predictSym(sym1,vocab,app_tag)
    if find==1:
        sym1=psym1
    else:
        i=0
        while True and i<len(psym1):
            print('Do you experience '+psym1[i].replace('_',' '))
            rep=input("")
            if str(rep)=='yes':
                sym1=psym1[i]
                break
            else:
                i=i+1

    print("Is there any other symtom Mr/Ms "+name+"  \n\t\t\t\t\t\t",end="=>")
    sym2=input("")
    psym2,find=predictSym(sym2,vocab,app_tag)
    if find==1:
        sym2=psym2
    else:
        i=0
        while True and i<len(psym2):
            print('Do you experience '+psym2[i].replace('_',' '))
            rep=input("")
            if str(rep)=='yes':
                sym2=psym2[i]
                break
            else:
                i=i+1
    
    #create patient symp list
    all_sym=(sym1,sym2)
    #predict possible intents
    intents=possible_intents(all_sym)
    stop=False
    print("Are you experiencing any ")
    for dis in intents:
        if stop==False:
            for sym in symVONintents(df_tr,dis):
                if sym not in all_sym:
                    print(clean_symp(sym)+' ?')
                    while True:
                        inp=input("")
                        if(inp=="yes" or inp=="no"):
                            break
                        else:
                            print("provide proper answers i.e. (yes/no) : ",end="")
                    if inp=="yes":
                        all_sym.append(sym)
                        dise=possible_intents(all_sym)
                        if len(dise)==1:
                            stop=True 
                            break
                    else:
                        continue
    return knn.predict(OHV(all_sym,all_symp_col)),all_sym
def chat_sp():
    a=True
    while a:
        name=getInfo()
        result,sym=main_sp(name)
        if result == None :
            ans3=input("can you specify more what you feel or tap q to stop the conversation")
            if ans3=="q":
                a=False
            else:
                continue

        else:
            print("you may have "+result[0])
            print(description_list[result[0]])
            an=input("how many day do you feel those symptoms ?")
            if calc_condition(sym,int(an))==1:
                print("you should take the consultation from doctor")
            else : 
                print('Take following precautions : ')
                for e in precautionDictionary[result[0]]:
                    print(e)
            print("do you need another medical consultation (yes or no)? ")
            ans=input()
            if ans!="yes":
                a=False
                print("!!!!! thanks for using ower application !!!!!! ")

chat_sp()