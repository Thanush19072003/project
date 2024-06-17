import json
import pandas as pd
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
import spacy

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

def preprocess(doc):
  """Preprocesses text by removing contractions, lemmatizing, and removing stopwords"""
  doc = doc.replace("'t", "not")
  nlp_doc = nlp(doc)
  tokens = []
  for token in nlp_doc:
    if token.text.lower() not in STOP_WORDS and token.text.isalpha():
      tokens.append(token.lemma_.lower())
  return ' '.join(tokens)

def preprocess_sent(sent):
  """Preprocesses a sentence by tokenizing, lemmatizing, and removing stopwords"""
  sent = sent.replace("'t", "not")
  tokens = nltk.word_tokenize(sent)
  return ' '.join([lemmatizer.lemmatize(w.lower()) for w in tokens if (w not in stopwords.words('english') and w.isalpha())])

def bag_of_words(tokenized_sentence, all_words):
  """Creates a bag-of-words representation of a sentence"""
  bag = np.zeros(len(all_words), dtype=np.float32)
  for idx, w in enumerate(all_words):
    if w in tokenized_sentence:
      bag[idx] = 1.0
  return bag

# Load data from JSON and CSV files (assuming they exist with proper structure)
with open('Medical_dataset/dieseas.json', 'r') as f:
  intents = json.load(f)

def load_data():
  """Loads data from CSV files (training data and symptom descriptions)"""
  df_tr = pd.read_csv('Medical_dataset/Training.csv')
  with open('Medical_dataset/symptom_Description.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    description_list = dict()
    for row in csv_reader:
      description_list[row[0]] = row[1]
  return df_tr, description_list

def get_severity_dict():
  """Loads severity data from CSV"""
  severityDictionary = dict()
  with open('Medical_dataset/symptom_severity.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    try:
      for row in csv_reader:
        severityDictionary[row[0]] = int(row[1])
    except:
      pass
  return severityDictionary

def get_precaution_dict():
  """Loads precaution data from CSV"""
  precautionDictionary = dict()
  with open('Medical_dataset/symptom_precaution.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
      precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]
  return precautionDictionary

def calc_condition(symptoms, days):
  """Calculates a score based on symptom severity and duration"""
  sum = 0
  for symptom in symptoms:
    sum += severityDictionary.get(symptom, 0)  # Handle missing values
  if (sum * days) / len(symptoms) > 13:
    return 1, "You should consult a doctor."
  else:
    return 0, "It might not be serious, but take precautions."

def predict_symptoms(symptom, vocab, app_tag):
  """Predicts possible diagnoses based on a symptom"""
  processed_symptom = preprocess_sent(symptom)
  bow = np.array(bag_of_words(processed_symptom, vocab))
  res = cosine_similarity(bow.reshape((1, -1)), df).reshape(-1)
  order = np.argsort(res)[::-1].tolist()
  possible_
