from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import csv
import spacy
import nltk
from textblob import TextBlob
from django.contrib import admin
from cities.models import Cities

from csvapp.models import CSVData
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from django.contrib.auth.models import User


nltk.download('stopwords')
from django.contrib.auth.models import User

def pakistan_drone_attacks(request):
    return render(request,'pakistan_drone_attacks.csv')
def provincewisedata(request):
    return render(request,'province_statepage.html')

## function for stats page like zameeen.com
def searchingarea(request):
    return render(request,'zameen.html')
def cardchecking(request):
    return render(request,'cardscheck.html')


# def karachi(request):
def generate_summary(content):
    # Concatenate all sentences from the content into a single text
    full_text = '\n'.join([' '.join(row) for row in content])

    # Tokenize the full text into sentences
    sentences = sent_tokenize(full_text)

    # Preprocess the sentences by removing stop words and punctuation
    stop_words = set(stopwords.words('english'))
    preprocessed_sentences = []
    for sentence in sentences:
        word_tokens = word_tokenize(sentence.lower())
        filtered_sentence = [word for word in word_tokens if word.isalnum() and word not in stop_words]
        preprocessed_sentences.append(' '.join(filtered_sentence))

    # Calculate the TF-IDF scores for the sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    scores = tfidf_matrix.sum(axis=1)

    # Find the index of the sentence with the highest TF-IDF score
    idx = scores.argmax()

    # Get the original sentence with the highest TF-IDF score
    summary = sentences[idx]

    return summary

def predict_crime_occurrence(content):
    # Concatenate all sentences from the content into a single text
    full_text = '\n'.join([' '.join(row) for row in content])

    # Perform Sentiment Analysis on the full text
    blob = TextBlob(full_text)
    sentiment_score = blob.sentiment.polarity

    # Determine whether the content indicates potential crime occurrence or not
    # For this example, we'll assume that a negative sentiment indicates potential crime.
    # You can adjust this threshold based on your specific use case and data.
    crime_prediction = "Not safe as residential point of view" if sentiment_score < 0 else "safe as residential point of view"

    return crime_prediction

def karachi(request):
    data = CSVData.objects.all()

    for item in data:
        if item.csv_name == 'karachi.csv':  # Replace 'csv2.csv' with the name of your second CSV file
            content = []
            extracted_entities = []  # Initialize inside the loop to reset for each row
            with open(item.file.path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file, delimiter=',')  # Adjust the delimiter as per your CSV
                for row in csv_reader:
                    content.append(row)
            item.content = content

            Locations = []
            Dates = []
            Times = []
            Persons = []
            Others = []
            Summaries = []  # List to store the summaries for each CSV file
            CrimePredictions = []  # List to store the crime predictions for each CSV file

            # Highlight date, time, and location entities in each row of the CSV content
            for row in content:
                for sentence in row:
                    # Tokenize the sentence
                    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                    sentences = tokenizer.tokenize(sentence)

                    # Process each sentence with spaCy
                    nlp = spacy.load('en_core_web_sm')
                    summary = generate_summary(content)
                    Summaries.append(summary)  # Append the summary to the list

                    # Perform Sentiment Analysis and get crime prediction
                    crime_prediction = predict_crime_occurrence(content)
                    CrimePredictions.append(crime_prediction)
                   
                    for sent in sentences:
                        doc = nlp(sent)
                        for ent in doc.ents:
                            extracted_entities.append({"text": ent.text, "label": ent.label_})

                            # Add entities to their respective variables
                            if ent.label_ == 'GPE':
                                Locations.append(ent.text)
                            elif ent.label_ == 'DATE':
                                Dates.append(ent.text)
                            elif ent.label_ == 'TIME':
                                Times.append(ent.text)
                            elif ent.label_ == 'PERSON':
                                Persons.append(ent.text)
                            else:
                                Others.append({"text": ent.text, "label": ent.label_})

            item.entities = extracted_entities

            # Render the template after processing all items
            return render(request, 'karachi.html', {
                'item': item,
                'Locations': Locations,
                'Dates': Dates,
                'Times': Times,
                'Persons': Persons,
                'Others': Others,
                'Summaries': Summaries,
                'CrimePredictions': CrimePredictions,
            })

    # Return an empty response if the CSV file is not found
    return render(request, 'karachi.html', {
        'data': data,
    })


## section of rawalpindi has been started
def rawalpindi(request):
    data = CSVData.objects.all()

    for item in data:
        if item.csv_name == 'rawalpindi.csv':  # Replace 'csv2.csv' with the name of your second CSV file
            content = []
            extracted_entities = []  # Initialize inside the loop to reset for each row
            with open(item.file.path, 'r', encoding='ISO-8859-1') as file:
                csv_reader = csv.reader(file, delimiter=',')  # Adjust the delimiter as per your CSV
                for row in csv_reader:
                    content.append(row)
            item.content = content

            Locations = []
            Dates = []
            Times = []
            Persons = []
            Others = []
            Summaries = []  # List to store the summaries for each CSV file
            CrimePredictions = []  # List to store the crime predictions for each CSV file

            # Highlight date, time, and location entities in each row of the CSV content
            for row in content:
                for sentence in row:
                    # Tokenize the sentence
                    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                    sentences = tokenizer.tokenize(sentence)

                    # Process each sentence with spaCy
                    nlp = spacy.load('en_core_web_sm')
                    summary = generate_summary(content)
                    Summaries.append(summary)  # Append the summary to the list

                    # Perform Sentiment Analysis and get crime prediction
                    crime_prediction = predict_crime_occurrence(content)
                    CrimePredictions.append(crime_prediction)
                   
                    for sent in sentences:
                        doc = nlp(sent)
                        for ent in doc.ents:
                            extracted_entities.append({"text": ent.text, "label": ent.label_})

                            # Add entities to their respective variables
                            if ent.label_ == 'GPE':
                                Locations.append(ent.text)
                            elif ent.label_ == 'DATE':
                                Dates.append(ent.text)
                            elif ent.label_ == 'TIME':
                                Times.append(ent.text)
                            elif ent.label_ == 'PERSON':
                                Persons.append(ent.text)
                            else:
                                Others.append({"text": ent.text, "label": ent.label_})

            item.entities = extracted_entities

            # Render the template after processing all items
            return render(request, 'rawalpindi.html', {
                'item': item,
                'Locations': Locations,
                'Dates': Dates,
                'Times': Times,
                'Persons': Persons,
                'Others': Others,
                'Summaries': Summaries,
                'CrimePredictions': CrimePredictions,
            })

    # Return an empty response if the CSV file is not found
    return render(request, 'rawalpindi.html', {
        'data': data,
    })

## sectoin of rawalpindi has been ended



## section of islamabad has been started
def islamabad(request):
    data = CSVData.objects.all()

    for item in data:
        if item.csv_name == 'islamabad.csv':  # Replace 'csv2.csv' with the name of your second CSV file
            content = []
            extracted_entities = []  # Initialize inside the loop to reset for each row
            with open(item.file.path, 'r', encoding='ISO-8859-1') as file:
                csv_reader = csv.reader(file, delimiter=',')  # Adjust the delimiter as per your CSV
                for row in csv_reader:
                    content.append(row)
            item.content = content

            Locations = []
            Dates = []
            Times = []
            Persons = []
            Others = []
            Summaries = []  # List to store the summaries for each CSV file
            CrimePredictions = []  # List to store the crime predictions for each CSV file

            # Highlight date, time, and location entities in each row of the CSV content
            for row in content:
                for sentence in row:
                    # Tokenize the sentence
                    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                    sentences = tokenizer.tokenize(sentence)

                    # Process each sentence with spaCy
                    nlp = spacy.load('en_core_web_sm')
                    summary = generate_summary(content)
                    Summaries.append(summary)  # Append the summary to the list

                    # Perform Sentiment Analysis and get crime prediction
                    crime_prediction = predict_crime_occurrence(content)
                    CrimePredictions.append(crime_prediction)
                   
                    for sent in sentences:
                        doc = nlp(sent)
                        for ent in doc.ents:
                            extracted_entities.append({"text": ent.text, "label": ent.label_})

                            # Add entities to their respective variables
                            if ent.label_ == 'GPE':
                                Locations.append(ent.text)
                            elif ent.label_ == 'DATE':
                                Dates.append(ent.text)
                            elif ent.label_ == 'TIME':
                                Times.append(ent.text)
                            elif ent.label_ == 'PERSON':
                                Persons.append(ent.text)
                            else:
                                Others.append({"text": ent.text, "label": ent.label_})

            item.entities = extracted_entities

            # Render the template after processing all items
            return render(request, 'islamabad.html', {
                'item': item,
                'Locations': Locations,
                'Dates': Dates,
                'Times': Times,
                'Persons': Persons,
                'Others': Others,
                'Summaries': Summaries,
                'CrimePredictions': CrimePredictions,
            })

    # Return an empty response if the CSV file is not found
    return render(request, 'islamabad.html', {
        'data': data,
    })

## section of islamabad has been ended


# section of gujranwala has been started
def gujranwala(request):
    data = CSVData.objects.all()

    for item in data:
        if item.csv_name == 'gujranwala.csv':  # Replace 'csv2.csv' with the name of your second CSV file
            content = []
            extracted_entities = []  # Initialize inside the loop to reset for each row
            with open(item.file.path, 'r', encoding='ISO-8859-1') as file:
                csv_reader = csv.reader(file, delimiter=',')  # Adjust the delimiter as per your CSV
                for row in csv_reader:
                    content.append(row)
            item.content = content

            Locations = []
            Dates = []
            Times = []
            Persons = []
            Others = []
            Summaries = []  # List to store the summaries for each CSV file
            CrimePredictions = []  # List to store the crime predictions for each CSV file

            # Highlight date, time, and location entities in each row of the CSV content
            for row in content:
                for sentence in row:
                    # Tokenize the sentence
                    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                    sentences = tokenizer.tokenize(sentence)

                    # Process each sentence with spaCy
                    nlp = spacy.load('en_core_web_sm')
                    summary = generate_summary(content)
                    Summaries.append(summary)  # Append the summary to the list

                    # Perform Sentiment Analysis and get crime prediction
                    crime_prediction = predict_crime_occurrence(content)
                    CrimePredictions.append(crime_prediction)
                   
                    for sent in sentences:
                        doc = nlp(sent)
                        for ent in doc.ents:
                            extracted_entities.append({"text": ent.text, "label": ent.label_})

                            # Add entities to their respective variables
                            if ent.label_ == 'GPE':
                                Locations.append(ent.text)
                            elif ent.label_ == 'DATE':
                                Dates.append(ent.text)
                            elif ent.label_ == 'TIME':
                                Times.append(ent.text)
                            elif ent.label_ == 'PERSON':
                                Persons.append(ent.text)
                            else:
                                Others.append({"text": ent.text, "label": ent.label_})

            item.entities = extracted_entities

            # Render the template after processing all items
            return render(request, 'gujranwala.html', {
                'item': item,
                'Locations': Locations,
                'Dates': Dates,
                'Times': Times,
                'Persons': Persons,
                'Others': Others,
                'Summaries': Summaries,
                'CrimePredictions': CrimePredictions,
            })

    # Return an empty response if the CSV file is not found
    return render(request, 'gujranwala.html', {
        'data': data,
    })
# section of gujranwala has been ended 


# section of faisalabad has been started
def faisalabad(request):
    data = CSVData.objects.all()

    for item in data:
        if item.csv_name == 'faisalabad.csv':  # Replace 'csv2.csv' with the name of your second CSV file
            content = []
            extracted_entities = []  # Initialize inside the loop to reset for each row
            with open(item.file.path, 'r', encoding='ISO-8859-1') as file:
                csv_reader = csv.reader(file, delimiter=',')  # Adjust the delimiter as per your CSV
                for row in csv_reader:
                    content.append(row)
            item.content = content

            Locations = []
            Dates = []
            Times = []
            Persons = []
            Others = []
            Summaries = []  # List to store the summaries for each CSV file
            CrimePredictions = []  # List to store the crime predictions for each CSV file

            # Highlight date, time, and location entities in each row of the CSV content
            for row in content:
                for sentence in row:
                    # Tokenize the sentence
                    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                    sentences = tokenizer.tokenize(sentence)

                    # Process each sentence with spaCy
                    nlp = spacy.load('en_core_web_sm')
                    summary = generate_summary(content)
                    Summaries.append(summary)  # Append the summary to the list

                    # Perform Sentiment Analysis and get crime prediction
                    crime_prediction = predict_crime_occurrence(content)
                    CrimePredictions.append(crime_prediction)
                   
                    for sent in sentences:
                        doc = nlp(sent)
                        for ent in doc.ents:
                            extracted_entities.append({"text": ent.text, "label": ent.label_})

                            # Add entities to their respective variables
                            if ent.label_ == 'GPE':
                                Locations.append(ent.text)
                            elif ent.label_ == 'DATE':
                                Dates.append(ent.text)
                            elif ent.label_ == 'TIME':
                                Times.append(ent.text)
                            elif ent.label_ == 'PERSON':
                                Persons.append(ent.text)
                            else:
                                Others.append({"text": ent.text, "label": ent.label_})

            item.entities = extracted_entities

            # Render the template after processing all items
            return render(request, 'faisalabad.html', {
                'item': item,
                'Locations': Locations,
                'Dates': Dates,
                'Times': Times,
                'Persons': Persons,
                'Others': Others,
                'Summaries': Summaries,
                'CrimePredictions': CrimePredictions,
            })

    # Return an empty response if the CSV file is not found
    return render(request, 'faisalabad.html', {
        'data': data,
    })
# section of faisalabad has been ended


# section of peshawar has been started
def peshawar(request):
    data = CSVData.objects.all()

    for item in data:
        if item.csv_name == 'peshawar.csv':  # Replace 'csv2.csv' with the name of your second CSV file
            content = []
            extracted_entities = []  # Initialize inside the loop to reset for each row
            with open(item.file.path, 'r', encoding='ISO-8859-1') as file:
                csv_reader = csv.reader(file, delimiter=',')  # Adjust the delimiter as per your CSV
                for row in csv_reader:
                    content.append(row)
            item.content = content

            Locations = []
            Dates = []
            Times = []
            Persons = []
            Others = []
            Summaries = []  # List to store the summaries for each CSV file
            CrimePredictions = []  # List to store the crime predictions for each CSV file

            # Highlight date, time, and location entities in each row of the CSV content
            for row in content:
                for sentence in row:
                    # Tokenize the sentence
                    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                    sentences = tokenizer.tokenize(sentence)

                    # Process each sentence with spaCy
                    nlp = spacy.load('en_core_web_sm')
                    summary = generate_summary(content)
                    Summaries.append(summary)  # Append the summary to the list

                    # Perform Sentiment Analysis and get crime prediction
                    crime_prediction = predict_crime_occurrence(content)
                    CrimePredictions.append(crime_prediction)
                   
                    for sent in sentences:
                        doc = nlp(sent)
                        for ent in doc.ents:
                            extracted_entities.append({"text": ent.text, "label": ent.label_})

                            # Add entities to their respective variables
                            if ent.label_ == 'GPE':
                                Locations.append(ent.text)
                            elif ent.label_ == 'DATE':
                                Dates.append(ent.text)
                            elif ent.label_ == 'TIME':
                                Times.append(ent.text)
                            elif ent.label_ == 'PERSON':
                                Persons.append(ent.text)
                            else:
                                Others.append({"text": ent.text, "label": ent.label_})

            item.entities = extracted_entities

            # Render the template after processing all items
            return render(request, 'peshawar.html', {
                'item': item,
                'Locations': Locations,
                'Dates': Dates,
                'Times': Times,
                'Persons': Persons,
                'Others': Others,
                'Summaries': Summaries,
                'CrimePredictions': CrimePredictions,
            })

    # Return an empty response if the CSV file is not found
    return render(request, 'peshawar.html', {
        'data': data,
    })
# section of peshawar has been ended



# section of multan has been started
def multan(request):
    data = CSVData.objects.all()

    for item in data:
        if item.csv_name == 'multan.csv':  # Replace 'csv2.csv' with the name of your second CSV file
            content = []
            extracted_entities = []  # Initialize inside the loop to reset for each row
            with open(item.file.path, 'r', encoding='ISO-8859-1') as file:
                csv_reader = csv.reader(file, delimiter=',')  # Adjust the delimiter as per your CSV
                for row in csv_reader:
                    content.append(row)
            item.content = content

            Locations = []
            Dates = []
            Times = []
            Persons = []
            Others = []
            Summaries = []  # List to store the summaries for each CSV file
            CrimePredictions = []  # List to store the crime predictions for each CSV file

            # Highlight date, time, and location entities in each row of the CSV content
            for row in content:
                for sentence in row:
                    # Tokenize the sentence
                    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                    sentences = tokenizer.tokenize(sentence)

                    # Process each sentence with spaCy
                    nlp = spacy.load('en_core_web_sm')
                    summary = generate_summary(content)
                    Summaries.append(summary)  # Append the summary to the list

                    # Perform Sentiment Analysis and get crime prediction
                    crime_prediction = predict_crime_occurrence(content)
                    CrimePredictions.append(crime_prediction)
                   
                    for sent in sentences:
                        doc = nlp(sent)
                        for ent in doc.ents:
                            extracted_entities.append({"text": ent.text, "label": ent.label_})

                            # Add entities to their respective variables
                            if ent.label_ == 'GPE':
                                Locations.append(ent.text)
                            elif ent.label_ == 'DATE':
                                Dates.append(ent.text)
                            elif ent.label_ == 'TIME':
                                Times.append(ent.text)
                            elif ent.label_ == 'PERSON':
                                Persons.append(ent.text)
                            else:
                                Others.append({"text": ent.text, "label": ent.label_})

            item.entities = extracted_entities

            # Render the template after processing all items
            return render(request, 'multan.html', {
                'item': item,
                'Locations': Locations,
                'Dates': Dates,
                'Times': Times,
                'Persons': Persons,
                'Others': Others,
                'Summaries': Summaries,
                'CrimePredictions': CrimePredictions,
            })

    # Return an empty response if the CSV file is not found
    return render(request, 'multan.html', {
        'data': data,
    })
# section of multan has been ended
# section of quetta has been started
def quetta(request):
    data = CSVData.objects.all()

    for item in data:
        if item.csv_name == 'quetta.csv':  # Replace 'csv2.csv' with the name of your second CSV file
            content = []
            extracted_entities = []  # Initialize inside the loop to reset for each row
            with open(item.file.path, 'r', encoding='ISO-8859-1') as file:
                csv_reader = csv.reader(file, delimiter=',')  # Adjust the delimiter as per your CSV
                for row in csv_reader:
                    content.append(row)
            item.content = content

            Locations = []
            Dates = []
            Times = []
            Persons = []
            Others = []
            Summaries = []  # List to store the summaries for each CSV file
            CrimePredictions = []  # List to store the crime predictions for each CSV file

            # Highlight date, time, and location entities in each row of the CSV content
            for row in content:
                for sentence in row:
                    # Tokenize the sentence
                    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                    sentences = tokenizer.tokenize(sentence)

                    # Process each sentence with spaCy
                    nlp = spacy.load('en_core_web_sm')
                    summary = generate_summary(content)
                    Summaries.append(summary)  # Append the summary to the list

                    # Perform Sentiment Analysis and get crime prediction
                    crime_prediction = predict_crime_occurrence(content)
                    CrimePredictions.append(crime_prediction)
                   
                    for sent in sentences:
                        doc = nlp(sent)
                        for ent in doc.ents:
                            extracted_entities.append({"text": ent.text, "label": ent.label_})

                            # Add entities to their respective variables
                            if ent.label_ == 'GPE':
                                Locations.append(ent.text)
                            elif ent.label_ == 'DATE':
                                Dates.append(ent.text)
                            elif ent.label_ == 'TIME':
                                Times.append(ent.text)
                            elif ent.label_ == 'PERSON':
                                Persons.append(ent.text)
                            else:
                                Others.append({"text": ent.text, "label": ent.label_})

            item.entities = extracted_entities

            # Render the template after processing all items
            return render(request, 'quetta.html', {
                'item': item,
                'Locations': Locations,
                'Dates': Dates,
                'Times': Times,
                'Persons': Persons,
                'Others': Others,
                'Summaries': Summaries,
                'CrimePredictions': CrimePredictions,
            })

    # Return an empty response if the CSV file is not found
    return render(request, 'quetta.html', {
        'data': data,
    })
# section of quetta haas been ended

def lahore(request):
    data = CSVData.objects.all()

    for item in data:
        if item.csv_name == 'lahore.csv':  # Replace 'csv2.csv' with the name of your second CSV file
            content = []
            extracted_entities = []  # Initialize inside the loop to reset for each row
            with open(item.file.path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file, delimiter=',')  # Adjust the delimiter as per your CSV
                for row in csv_reader:
                    content.append(row)
            item.content = content

            Locations = []
            Dates = []
            Times = []
            Persons = []
            Others = []
            Summaries = []  # List to store the summaries for each CSV file
            CrimePredictions = []  # List to store the crime predictions for each CSV file

            # Highlight date, time, and location entities in each row of the CSV content
            for row in content:
                for sentence in row:
                    # Tokenize the sentence
                    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                    sentences = tokenizer.tokenize(sentence)

                    # Process each sentence with spaCy
                    nlp = spacy.load('en_core_web_sm')
                    summary = generate_summary(content)
                    Summaries.append(summary)  # Append the summary to the list
                    # Summaries = summary
                    # Perform Sentiment Analysis and get crime prediction
                    crime_prediction = predict_crime_occurrence(content)
                    CrimePredictions.append(crime_prediction)
                   
                    for sent in sentences:
                        doc = nlp(sent)
                        for ent in doc.ents:
                            extracted_entities.append({"text": ent.text, "label": ent.label_})

                            # Add entities to their respective variables
                            if ent.label_ == 'GPE':
                                Locations.append(ent.text)
                            elif ent.label_ == 'DATE':
                                Dates.append(ent.text)
                            elif ent.label_ == 'TIME':
                                Times.append(ent.text)
                            elif ent.label_ == 'PERSON':
                                Persons.append(ent.text)
                            else:
                                Others.append({"text": ent.text, "label": ent.label_})

            item.entities = extracted_entities

            # Render the template after processing all items
            return render(request, 'lahore.html', {
                'item': item,
                'Locations': Locations,
                'Dates': Dates,
                'Times': Times,
                'Persons': Persons,
                'Others': Others,
                'Summaries': Summaries,
                'CrimePredictions': CrimePredictions,
            })

    # Return an empty response if the CSV file is not found
    return render(request, 'lahore.html', {
        'data': data,
    })



def myhomepage(request):
    return render(request, "myhomepage.html")


def header(request):
    return render(request, "header.html")

def carousel(request):
    return render(request, "carousel_pane.html")

def registercrime(request):
    return render(request, "register_crime.html")

def footer(request):
    return render(request, "footer.html")

def flipcard(request):
    return render ( request, 'flip_cards.html')

from django.shortcuts import render
from django.core import serializers
import json

from django.http import JsonResponse

from django.http import JsonResponse
import json
from django.core import serializers

from django.http import JsonResponse


def citiesdata(request):
    cities = Cities.objects.all()
    print(cities)
    context = {'cities': cities}
    return render(request, 'hover_cards.html', context)


def home(request):
    return render(request,'home.html')


def homepage(request):
    user_count = User.objects.count()  # Get the user count
    return render(request, 'homepage.html', {'user_count': user_count})


def iconsmarquee(request):
    return render (request, "icons_marquee.html")


def hometemplate(request):
  
    return render(request, 'homepage_template/index.html')


   


from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
# Create your views here.
@login_required(login_url='login')
def HomePage(request):
    return render (request,'home_template.html')

def SignupPage(request):
    if request.method=='POST':
        uname=request.POST.get('username')
        email=request.POST.get('email')
        pass1=request.POST.get('password1')
        pass2=request.POST.get('password2')

        print(uname,email,pass1,pass2)

        if pass1!=pass2:
            return HttpResponse("Your password and confrom password are not Same!!")
        else:

            my_user=User.objects.create_user(uname,email,pass1)
            my_user.save()
            return redirect('login')

    return render (request,'signup.html')

def LoginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            return redirect('home')
        else:
            return HttpResponse ("Username or Password is incorrect!!!")

    return render (request,'signin.html')

def LogoutPage(request):
    logout(request)
    return redirect('login')




def hovercard(request):
   
    return render(request, 'hover_cards.html', context)




from django.shortcuts import render
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import json
from statisticalanalysis.models import StatisticalAnalysis

def crimestats(request):
    
    figure_json1 = None
    figure_json2 = None
    figure_json3 = None
    figure_json4 = None
    figure_json5 = None
    figure_json6 = None
    figure_json7 = None
    figure_json8 = None
    figure_json9 = None
    figure_json10 = None


    data = StatisticalAnalysis.objects.all()
    for item in data:
        if item.csv_name == 'crimestats.csv':
            df = pd.read_csv(item.file.path)
            
            # Check if the dataframe has the required columns
            if all(col in df.columns for col in ['year', 'division', 'crime type', 'cases']):
                    
                ################### Bar Plot Of Different Crime Categories starts here #################

                fig1 = px.bar(df, x='division', y='cases', title='Total Number of Cases for Each Crime Type', color='crime type', hover_data=['year', 'district'])
                fig1.update_layout(title='Crime Stats In KPK', title_x=0.45, title_y=0.0)
                # Set the plot and paper background to transparent
                fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                #################TBar Plot Of Different Crime Categories ends here ###############################
                
                #######################   sunburst plot of all record starts here #################

                fig2 = px.sunburst(
                    data_frame=df, 
                    path=["year", 'division', 'crime type'], 
                    values="cases",
                    width=1000, 
                    height=500,
                  
                     
                    color="crime type",
                    maxdepth=3,
                    hover_data=["cases", "year", 'crime type']
                )
                fig2.update_layout(title='Sunburst Plot of Crimes', title_x=0.5, title_y=0.0)
                # Set the plot and paper background to transparent
  

                 #######################   sunburst plot of all record ends here #################


            
               ########################  crime trend over  years starts here #####################

               ########################  crime trend over  years ends here #####################
              

                # Convert the year column to string to treat it as categorical data
                # df['year'] = df['year'].astype(str)

                # # Group by year and sum the cases
                # yearly_cases = df.groupby('year')['cases'].sum().reset_index()

                # # Create the bubble chart
                # fig = px.scatter(yearly_cases, x='year', y='cases', size='cases', color='year',
                #                 title='Total Crime Cases Over the Years',
                #                 labels={'cases': 'Total Number of Cases', 'year': 'Year'},
                #                 hover_data={'cases': ':,'})

                # # Update hovertemplate to show total cases
                # fig.update_traces(hovertemplate='Year: %{x}<br>Total Cases: %{y:,}')

               
                # # Set other layout properties
                # fig.update_layout(plot_bgcolor='white', template="ggplot2", height=500,
                #     width=1000)
                # fig.update_xaxes(showgrid=False)
                # fig.update_yaxes(showgrid=False)

                # # Show the plot
                # fig3 = fig

                # Convert the year column to string to treat it as categorical data
                df['year'] = df['year'].astype(str)

                # Group by year and sum the cases
                yearly_cases = df.groupby('year')['cases'].sum().reset_index()
                # Create the bubble chart
                fig = px.scatter(yearly_cases, x='year', y='cases', size='cases', color='year',
                                title='Total Crime Cases Over the Years',
                                labels={'cases': 'Total Number of Cases', 'year': 'Year'},
                                hover_data={'cases': ':,'},
                                size_max=60)

                # Update hovertemplate to show total cases
                fig.update_traces(hovertemplate='Year: %{x}<br>Total Cases: %{y:,}')

                # Set the image as a background
             
                # Set the plot and paper background to transparent
                fig.update_layout(height=500,width=1000,template='plotly_dark')

                # Set other layout properties
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
                
                fig3 = fig

            ############################ Percent of crime trend over  years starts here    ######################
                # Convert the year column to string to treat it as categorical data
                df['year'] = df['year'].astype(str)

                # Group by year and sum the cases
                yearly_cases = df.groupby('year')['cases'].sum().reset_index()

                # Create the pie chart
                fig = px.pie(yearly_cases, names='year', values='cases',
                            title='Total Crime Cases Over the Years',
                            labels={'cases': 'Total Number of Cases', 'year': 'Year'},
                            hover_data={'cases': True}, height=500,
                    width=1000)

                # Update hovertemplate to show total cases
                fig.update_traces(hovertemplate='Year: %{label}<br>Total Cases: %{value:,}<extra></extra>')

                # Set the plot and paper background to transparent
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

                # Show the plot
                fig4 = fig
            ############################ Percent of crime trend over  years ends here    ######################

            ################################  most affected division starts here ####################################
            # Group the data by division and year, and sum the cases
                df_grouped = df.groupby(['division'])['cases'].sum().reset_index()

                # Sort the DataFrame in descending order based on the number of cases
                df_sorted = df_grouped.sort_values(by='cases', ascending=True)

                # Find the year with the highest number of cases for each division
                idx = df_grouped.groupby('division')['cases'].idxmax()
                df_max_cases = df_grouped.loc[idx]

                # Create the bar plot
                fig = px.bar(df_max_cases, x='division', y='cases', color='cases',
                            hover_data=['cases'],
                            labels={'cases': 'Number of Cases', 'division': 'Division'},
                            title='Cases in Each Division',
                            height=500,
                            width=1000)
                # Set the plot and paper background to transparent
                fig.update_layout(template="plotly_dark")

                # Show the plot
                fig5 = fig
            ################################  most affected division ends here ####################################

            ################################ most affected district starts here ##########################
                # Aggregate data
                agg_data = df.groupby('district')['cases'].sum().reset_index()

                # Create a horizontal bar plot
                fig = px.bar(
                    agg_data,
                    x='cases',
                    y='district',
                    orientation='h',
                    color='district',
                    title="Total Crime Count by District",
                    hover_data={'cases': ':,'}  # Format hover data as a simple number

                )

                # Hide the legend
                fig.update_layout(showlegend=False, height=500,
                    width=1000)
                
                # Set the plot and paper background to transparent
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                
                fig6 = fig
            ################################ most affected district ends here ##########################

            ##############################  most occuring crime starts here #########################################
                # Group the data by crime type and sum the cases
                df_grouped = df.groupby('crime type')['cases'].sum().reset_index()

                # Sort the DataFrame in descending order based on the number of cases
                df_sorted = df_grouped.sort_values(by='cases', ascending=False)

                # Create the funnel chart
                fig = px.funnel(df_sorted, x='cases', y='crime type',
                                title='Crime Cases Funnel')

                # Update the layout if needed
                fig.update_layout(showlegend=False ,height=500,
                    width=1000)

                # Set the plot and paper background to transparent
                fig.update_layout(template="plotly_dark")
                
                # Show the plot
                fig7 = fig

            ##############################  most occuring crime ends here #########################################

           
            ################################# Total Number of Cases for Each Division by Year starts here #########################
                # year , division comparisons 
                # Group the data by division and year, and sum the cases
                df_grouped = df.groupby(['division', 'year'])['cases'].sum().reset_index()

                # Create a bar plot
                fig = px.bar(
                    df_grouped,
                    x='division',
                    y='cases',
                    color='year',
                    title="Total Number of Cases for Each Division by Year",
                    labels={'cases': 'Number of Cases', 'year': 'Year', 'division': 'Division'},
                    hover_data={'cases': ':,'},
                    barmode='group',
                    height=500,
                    width=1000
                )

                # Set the plot and paper background to transparent
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

                # Show the plot
                fig8 = fig
            ################################# Total Number of Cases for Each Division by Year ends here #########################
            

            ################################ Crime Counts by Crime Type Over Years starts here ##################################

                # Aggregate data by Year and CrimeType
                agg_data = df.groupby(['year', 'crime type'])['cases'].sum().reset_index()

                # Create a scatter bubble chart
                fig = px.scatter(
                    agg_data,
                    x='cases',
                    y='year',
                    size='cases',  # Size of the bubbles will be based on the number of cases
                    color='crime type',  # Different colors for each crime type
                    title='Crime Counts by Crime Type Over Years',
                    hover_data={'cases': ':,'},  # Format hover data as a simple number
                    template='plotly_dark'
                )

                # Update layout to add axis labels
                fig.update_layout(
                    xaxis_title='Crime Count',
                    yaxis_title='Year',
                    height=500,
                    width=900
                )

             

                fig9 = fig
            ################################ Crime Counts by Crime Type Over Years ends here ##################################


            ############################## Crime Counts by Division Over Years starts here ##################################
                # Aggregate data by Year and Division
                agg_data = df.groupby(['year', 'division'])['cases'].sum().reset_index()

                # Create a line chart
                fig = px.line(
                    agg_data,
                    x='year',  # The x-axis will be the 'year'
                    y='cases',  # The y-axis will be the sum of 'cases'
                    color='division',  # Different lines for each 'division'
                    title='Crime Counts by Division Over Years',
                    markers=True,  # Add markers to the line chart
                    line_shape='linear',  # Use linear line shape
                    hover_data={'cases': ':,'},  # Format hover data as a simple number
                   
                    height=500,
                    width=1000
                )

                # Update layout to add axis labels
                fig.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Crime Count',
                )
                                # Set the plot and paper background to transparent
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')


                fig10 = fig
            ############################## Crime Counts by Division Over Years ends here ##################################

                
                # Convert the figure to JSON format using Plotly's built-in method
                figure_json1 = fig1.to_json()
                figure_json2 = fig2.to_json()
                figure_json3 = fig3.to_json()
                figure_json4 = fig4.to_json()
                figure_json5 = fig5.to_json()
                figure_json6 = fig6.to_json()
                figure_json7 = fig7.to_json()
                figure_json8 = fig8.to_json()
                figure_json9 = fig9.to_json()
                figure_json10 = fig10.to_json()


                
            else:
                return HttpResponse("The uploaded CSV is missing some required columns.")

    return render(request, 'stats.html', {'figure1': figure_json1,'figure2': figure_json2,'figure3': figure_json3,'figure4': figure_json4,'figure5': figure_json5,'figure6': figure_json6,'figure7': figure_json7,'figure8': figure_json8,'figure9': figure_json9,'figure10': figure_json10})






##################################### pakistan drone attacks file content starts  here ####################

def pakistan_drone_attacks(request):
    
    figure_json1 = None
    figure_json2 = None
    figure_json3 = None
    figure_json4 = None
    figure_json5 = None
    figure_json6 = None
    figure_json7 = None
    figure_json8 = None
    figure_json9 = None
    figure_json10 = None
    figure_json11 = None
    figure_json12 = None
    figure_json13 = None


    data = StatisticalAnalysis.objects.all()
    for item in data:
        if item.csv_name == 'Pakistan_drone_attacks.csv':
            df = pd.read_csv(item.file.path)
            print(df.head())
            
            # Check if the dataframe has the required columns
            if all(col in df.columns for col in ['Date', 'Time', 'Location', 'City', 'Province', 'No of Strike','Al-Qaeda','Taliban', 'Civilians Min', 'Civilians Max','Foreigners Min', 'Foreigners Max', 'Total Died Min', 'Total Died Mix','Injured Min', 'Injured Max', 'Women/Children','Special Mention (Site)', 'Comments', 'References', 'Longitude','Latitude', 'Temperature(C)', 'Temperature(F)', 'Total Died Max','year', 'customDate']):
                    
                
               ################################  days killing start  session ##########################################
                # The Date format seems to be 'weekday- month day- year'
                # We use this format to parse the date and then extract the day of the week
                df['Day of Week'] = pd.to_datetime(df['Date'], format='%A- %B %d- %Y', errors='coerce').dt.day_name()

                # Count the number of strikes on each day of the week
                day_counts = df['Day of Week'].value_counts()

                # Create the bar chart using Plotly Express
                fig = px.bar(day_counts,
                            x=day_counts.index,
                            y=day_counts.values,
                            labels={'x': 'Day of the Week', 'y': 'Number of Strikes'},
                            title='Number of Drone Strikes by Day of the Week')

                fig.update_layout(
                    template = 'plotly_dark'
                )

                # Show the plot
                fig1= fig
               ####################################### days killing  END SESSION ############################


               ################################# monthly killings start session ##############################
                # The Date format seems to be 'weekday- month day- year'
                # We use this format to parse the date and then extract the month
                df['Month'] = pd.to_datetime(df['Date'], format='%A- %B %d- %Y', errors='coerce').dt.month_name()

                # Count the number of strikes in each month
                month_counts = df['Month'].value_counts()

                # Create the bar chart using Plotly Express
                fig = px.bar(month_counts,
                            x=month_counts.index,
                            y=month_counts.values,
                            labels={'x': 'Month', 'y': 'Number of Strikes'},
                            title='Number of Drone Strikes by Month')

                fig.update_layout(
                    template = 'plotly_dark'
                )

                # Show the figure
                fig2 = fig
               ################################ monthly killings end session ################################



                ############################ hoursly killings starts here #################################

                # Extracting the hour from the 'Time' column
                # Assuming the time format is 'HH:MM' in the 'Time' column
                df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour

                # Dropping rows where time is NaN or couldn't be converted
                df = df.dropna(subset=['Hour'])

                # Converting Hour to integer
                df['Hour'] = df['Hour'].astype(int)

                # Create the histogram using Plotly Express
                fig = px.histogram(df,
                                x='Hour',
                                nbins=24,  # 24 bins for 24 hours
                                color='Hour',  # Assign different colors based on Hour
                                labels={'Hour': 'Hour of the Day'},
                                title='Histogram of Drone Strikes by Hour of the Day')

                fig.update_layout(
                    template = 'plotly_dark',
                    xaxis_title='Hour of the Day',
                    yaxis_title='Number of Strikes',
                    coloraxis_colorbar=dict(
                        title='Hour of the Day'
                    )
                )

                # Show the figure
                fig3=fig


                ############################## hourly killings ends here ######################################



                ############################# People Killed & Injured per Year In Last 12-Years  starts here ###############
                import numpy as np
                dfKill = pd.DataFrame(df, columns=['year', 'Al-Qaeda', 'Taliban',
                                                'Civilians Min', 'Civilians Max',
                                                'Foreigners Min', 'Foreigners Max',
                                                'Total Died Min', 'Total Died Max',
                                                'Injured Min', 'Injured Max', 'No of Strike'])

                dfKill['Civilians'] = ((dfKill['Civilians Min'] + dfKill['Civilians Max'])/2).apply(np.ceil).astype('int64')
                dfKill['Foreigners'] = ((dfKill['Foreigners Min'] + dfKill['Foreigners Max'])/2).apply(np.ceil).astype('int64')
                dfKill['Total-Injured'] = ((dfKill['Injured Min'] + dfKill['Injured Max'])/2).apply(np.ceil).astype('int64')
                dfKill['Total-Killed'] = ((dfKill['Total Died Min'] + dfKill['Total Died Max'])/2).apply(np.ceil).astype('int64')

                dfKill.rename( columns={'No of Strike' : 'no-of-strike'}, inplace=True)
                dfKill['no-of-strike'] = (dfKill['no-of-strike']).apply(np.ceil).astype('int64')

                dfKill = dfKill.drop(['Civilians Min', 'Civilians Max', 'Foreigners Min', 'Foreigners Max',
                                    'Total Died Min', 'Total Died Max', 'Injured Min', 'Injured Max'], axis=1)
                #dfKill
                dfKillbyYear = dfKill.groupby(['year'], as_index=False).sum().sort_values('year', ascending=False)
                # Assuming dfKillbyYear is already defined

                # Reshape the DataFrame from wide to long format
                df_long = pd.melt(dfKillbyYear, id_vars='year', var_name='Category', value_name='Numbers')

                # Create the line plot
                fig = px.line(df_long, x='year', y='Numbers', color='Category',
                            labels={'Numbers': 'Numbers', 'year': 'YEARS'},
                            title="Killed & Injured Per Year")

                # Update layout
                fig.update_layout(xaxis=dict(tickmode='array',
                                            tickvals=list(range(dfKillbyYear['year'].min(), dfKillbyYear['year'].max()+1))),
                                xaxis_title="YEARS",
                                yaxis_title="Numbers",
                                legend_title_text='Category',
                                font=dict(size=16))
                # Set the plot and paper background to transparent
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                # Show the plot
                fig4 = fig

                ######################### People Killed & Injured per Year In Last 12-Years  ends here  #######################
                
            ######################### Number of Drone Attack, Taliban & Al-Qaeda Targeted Per Year starts here  #####################

                dfStriker = dfKillbyYear.drop(['Civilians', 'Foreigners', 'Total-Injured', 'Total-Killed'], axis=1)

                # Assuming dfStriker is already defined and contains the columns 'year', 'no-of-strike', 'Taliban', 'Al-Qaeda'
                # Reshape the DataFrame from wide to long format
                df_long = pd.melt(dfStriker, id_vars='year', var_name='Category', value_name='Numbers')

                # Create the line plot
                fig = px.line(df_long, x='year', y='Numbers', color='Category',
                            labels={'Numbers': 'Numbers', 'year': 'YEARS'},
                            title="Number of Drone Attack, Taliban & Al-Qaeda Targeted Per Year")

                # Update layout for customizations similar to the matplotlib plot
                fig.update_layout(xaxis=dict(tickmode='array',
                                            tickvals=list(range(dfStriker['year'].min(), dfStriker['year'].max()+1))),
                                xaxis_title="YEARS",
                                yaxis_title="Numbers",
                                legend_title_text='Category',
                                template='plotly_dark',
                                font=dict(size=16),
                                width=1000,
                                height=600)

                # Show the plot
                fig5= fig

            ################ Number of Drone Attack, Taliban & Al-Qaeda Targeted Per Year ends here #####################


############################## Number of Drone Attacks, Al-Qaeda & Taliban Targeted starts here ################

                # Create the bar plot using Plotly Express
                fig = px.bar(dfStriker, x='year',
                            y=['no-of-strike', 'Taliban', 'Al-Qaeda'],
                            labels={'value': 'Numbers', 'variable': 'Category', 'year': 'Years'},
                            barmode='group')

                # Update layout for customizations
                fig.update_layout(title={'text': "Number of Drone Attacks, Al-Qaeda & Taliban Targeted", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                                title_font_size=24,
                                xaxis_title="Years",
                                yaxis_title="Numbers",
                                legend_title="Category",
                                font=dict(size=15),
                                width=1100,
                                height=600,
                                xaxis_tickangle=-30)
                # Set the plot and paper background to transparent
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                # Show the plot
                fig6 = fig

################################   Number of Drone Attacks, Al-Qaeda & Taliban Targeted ends here ###################


####################################  Women & Children Involved starts here  #################################### 

                # Fill Empty with 'n' and replace 'y' with 1 and 'n' with 0
                # Where y & 1 = Yes,    n & 0 = No
                df['Women/Children'] = df['Women/Children'].fillna('n')
                df['Women/Children'] = df['Women/Children'].replace('y', 1)
                df['Women/Children'] = df['Women/Children'].replace('n', 0)
                df['Women/Children'] = df['Women/Children'].astype('int64')
                # Calculate the counts for each category in 'Women/Children'
                count_df = df['Women/Children'].value_counts().reset_index()
                count_df.columns = ['Women/Children', 'count']

                # Create the bar plot using the count DataFrame
                fig = px.bar(count_df, x='Women/Children', y='count', text='count')

                # Update layout for customizations
                fig.update_layout(title='Number of Women/Children Involved in Drone Attacks',
                                xaxis_title='Women/Children [0=No, 1=Yes]',
                                yaxis_title='Numbers',
                                font=dict(size=15),
                                title_font_size=18,
                                template= 'plotly_dark')


                # Show the plot
                fig7=fig

        ###################################   Women & Children Involved ends here   ####################################


            ################################ Number of Strikes by Year starts here ########################

                dfWomenChild = pd.DataFrame(df, columns=['year', 'Women/Children', 'No of Strike']).astype('int64')
                dfWomenChild = dfWomenChild.rename(columns={'No of Strike': 'no-of-strike'})
                dfWomenChildbyYear = dfWomenChild[dfWomenChild['Women/Children'] != 0 ].groupby(['year', 'Women/Children'], as_index=False).sum().sort_values('year', ascending=False)

                # Assuming dfWomenChildbyYear is already defined and contains the columns 'year', 'Women/Children', and 'no-of-strike'
                fig = px.bar(dfWomenChildbyYear, x='year', y='no-of-strike', color='year',
                            barmode='group', labels={'no-of-strike': 'Number of Strikes'})

                # Update layout for customizations
                fig.update_layout(title='Number of Strikes by Year',
                                xaxis_title='Year',
                                yaxis_title='Number of Strikes',
                                legend_title='Women/Children',
                                font=dict(size=15),
                                title_font_size=18,
                                height=500,
                                width=1000)

                # Show the plot
                fig8=fig

            ############################ Number of Strikes by Year ends here ###########################

            ################################# Drone Attacks on Timeline  starts here #############################
                import plotly.graph_objs as go
                trace = go.Scatter(
                    x = dfKillbyYear['year'],
                    y = dfKillbyYear['no-of-strike'],
                    mode = 'lines+markers'
                )
                fig = go.Figure(data=trace)
                fig.update_xaxes(title='Year')
                fig.update_yaxes(title='Number of Attacks')
                fig.update_layout(title_text='Drone Attacks Timeline', title_x=0.5, title_font_size=25, template='plotly_dark')

                # Show the plot
                fig9=fig

            ################################ Drone Attacks on Timeline ends here #################################


                ############################## Drone Attacks in Bush & Obama Tenure starts here  ##############################

                df['customDate'] = pd.to_datetime(df['Date'])
                dfTenure = pd.DataFrame()
                dfTenure['year'] = df['year'].unique()
                dfTenure['GWBush'] = 0
                dfTenure['BObama'] = 0
                dfTenure['DTrump'] = 0

                for x, xRow in dfTenure.iterrows():

                    if(xRow['year'] <= 2009):
                        getTot = df[(df['customDate'] < '2009-01-20') & (df['year'] == xRow['year'])]['No of Strike'].sum()
                        dfTenure.at[x, 'GWBush'] = getTot
                        dfTenure.at[x, 'BObama'] = dfTenure.at[x, 'BObama'] + 0
                        dfTenure.at[x, 'DTrump'] = dfTenure.at[x, 'DTrump'] + 0

                    if(xRow['year'] >= 2009):
                        getTot = df[(df['customDate'] >= '2009-01-20') & (df['customDate'] < '2017-01-20') & (df['year'] == xRow['year']) ]['No of Strike'].sum()
                        dfTenure.at[x, 'GWBush'] = dfTenure.at[x, 'GWBush'] + 0
                        dfTenure.at[x, 'DTrump'] = dfTenure.at[x, 'DTrump'] + 0
                        dfTenure.at[x, 'BObama'] = getTot

                    if (xRow['year'] >= 2017):
                        getTot = df[(df['customDate'] >= '2017-01-20') & (df['year'] == xRow['year']) ]['No of Strike'].sum()
                        dfTenure.at[x, 'GWBush'] = dfTenure.at[x, 'GWBush'] + 0
                        dfTenure.at[x, 'BObama'] = dfTenure.at[x, 'GWBush'] + 0
                        dfTenure.at[x, 'DTrump'] = getTot
                # Assuming dfTenure is your DataFrame and it contains 'year', 'GWBush', 'BObama', 'DTrump' columns
                fig = px.bar(dfTenure, x='year',
                            y=['GWBush', 'BObama', 'DTrump'],
                            barmode='group',
                            labels={'value': 'Number of Drone Attacks', 'variable': 'President', 'year': 'Years'})

                # Update layout for customizations
                fig.update_layout(title='Number of Drone Attacks Comparison in Tenure of US Presidents',
                                xaxis_title='Years',
                                yaxis_title='Number of Drone Attacks',
                                legend_title="President",
                                font=dict(size=15),
                                title_font_size=22,
                                template= 'plotly_dark',
                                width=900,
                                height=600,
                                xaxis_tickangle=-30)

                # Show the plot
                fig10=fig

        ################## Drone Attacks in Bush & Obama Tenure ends here #######################################




#####################  Drone Attacks in Bush & Obama Tenure - line chart comparison starts here    ##########################

                # compasrins of attacks on US president era

                # Assuming dfTenure is already defined and contains the columns 'year', 'GWBush', 'BObama', 'DTrump'
                # Reshaping the DataFrame from wide to long format
                df_long = dfTenure.melt(id_vars='year', var_name='President', value_name='Number of Drone Attacks')

                # Create the line plot
                fig = px.line(df_long, x='year', y='Number of Drone Attacks', color='President',
                            labels={'Number of Drone Attacks': 'Number of Drone Attacks', 'year': 'YEARS'},
                            title="Drone Attacks Comparison in Tenure of US Presidents")

                # Update layout for customizations
                fig.update_layout(

                                    template='ggplot2',
                                    plot_bgcolor='#23272c',
                                    paper_bgcolor='#23272c',
                                    xaxis_title="YEARS",
                                    yaxis_title="Number of Drone Attacks",
                                    title_font=dict(size=24, color='#FFFFFF'),
                                    font=dict(color='#FFFFFF'),
                                    xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='#FFFFFF'),
                                    yaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='#FFFFFF'),

                )

                # Show the plot
                fig11=fig

########################## Drone Attacks in Bush & Obama Tenure - line chart comparison ends here #######################


############################  Number of Drone Attacks Per City/Location  starts here  #############################

                import plotly.graph_objs as go
                from plotly.offline import iplot
                import numpy as np
                import itertools
                # Number of Drone Attacks Per City/Location
                attackpercity = pd.DataFrame(columns=['city', 'total_attack'])
                attackpercity['city'] = df['City'].unique().tolist()
                attackpercity['total_attack'] = attackpercity['total_attack'].fillna('0').astype(np.int64)
                for x, xRow in attackpercity.iterrows():
                    tot_attack = df[df['City'] == xRow['city']]['No of Strike'].sum()
                    attackpercity.at[x, 'total_attack'] = tot_attack


                # Assuming attackpercity is your DataFrame
                # Get a list of unique cities
                unique_cities = attackpercity['city'].unique()

                # Create a color map, cycling through the Plotly colors if necessary
                color_palette = px.colors.qualitative.Plotly
                city_colors = {city: color for city, color in zip(unique_cities, itertools.cycle(color_palette))}

                figCity = [go.Bar(y=attackpercity['total_attack'],
                                x=attackpercity['city'],

                                textposition='outside',
                                width=[0.7] * len(attackpercity),  # Set uniform width for all bars
                                marker=dict(color=[city_colors[city] for city in attackpercity['city']],  # Map colors
                                            line_color='black',
                                            line_width=2))]

                layout = go.Layout(title='Drone Attacks Per City',
                                xaxis=dict(title='Cities Name'),
                                yaxis=dict(title='Drone Attacks'),
                                width=900,
                                height=600,
                                template='plotly_white')

                fig_plot = go.Figure(data=figCity, layout=layout)
                fig12=fig_plot  

############################      Number of Drone Attacks Per City/Location ends here  #########################



            #################################### Civilians vs Terrorist  starts here ##########################

                dfCivilians = pd.DataFrame(df, columns=['year', 'Al-Qaeda', 'Taliban',
                                                        'Civilians Min', 'Civilians Max',
                                                        'Foreigners Min', 'Foreigners Max'])

                dfCivilians['Civilians'] = (((dfCivilians['Civilians Min'] + dfCivilians['Civilians Max'])/2) +
                                            ((dfCivilians['Foreigners Min'] + dfCivilians['Foreigners Max'])/2)).apply(np.ceil).astype('int64')
                dfCivilians = dfCivilians.drop(['Civilians Min', 'Civilians Max', 'Foreigners Min', 'Foreigners Max'], axis=1)

                dfCivilians['Terrorist'] = (dfCivilians['Al-Qaeda'] + dfCivilians['Taliban']).apply(np.ceil).astype('int64')
                #dfCivilians
                dfCiviliansbyYear = dfCivilians.groupby(['year'], as_index=False).sum().sort_values('year', ascending=True)
                # Assuming dfCiviliansbyYear is your DataFrame and it contains 'year', 'Civilians', 'Terrorist' columns
                fig = px.bar(dfCiviliansbyYear, x='year',
                            y=['Civilians', 'Terrorist'],
                            barmode='group',
                            labels={'value': 'Numbers', 'variable': 'Category', 'year': 'Years'})

                # Update layout for customizations
                fig.update_layout(title='Civilians vs Terrorist',
                                xaxis_title='Years',
                                yaxis_title='Numbers',
                                legend_title="Category",
                                font=dict(size=15),
                                title_font_size=24,
                                template= 'ggplot2',
                                width=900,
                                height=600,
                                xaxis_tickangle=-30)

                # Show the plot
                fig13 = fig

            ############################   Civilians vs Terrorist  ends here ###########################





                # Convert the figure to JSON format using Plotly's built-in method
                figure_json1 = fig1.to_json()
                figure_json2 = fig2.to_json()
                figure_json3 = fig3.to_json()
                figure_json4 = fig4.to_json()
                figure_json5 = fig5.to_json()
                figure_json6 = fig6.to_json()
                figure_json7 = fig7.to_json()
                figure_json8 = fig8.to_json()
                figure_json9 = fig9.to_json()
                figure_json10 = fig10.to_json()
                figure_json11 = fig11.to_json()
                figure_json12 = fig12.to_json()
                figure_json13 = fig13.to_json()
               


                
            else:
                return HttpResponse("The uploaded CSV is missing some required columns.")

    return render(request, 'pakistan_drone_attacks.html', {'figure1': figure_json1,'figure2': figure_json2,'figure3': figure_json3,'figure4': figure_json4,'figure5':figure_json5,'figure6':figure_json6,'figure7':figure_json7,'figure8':figure_json8,'figure9':figure_json9,'figure10':figure_json10,'figure11':figure_json11,'figure12':figure_json12,'figure13':figure_json13})

            
 

############################### PAKISTNA drone attakcs content session ends here ###########################



######################################  PAKISTAN SUICIDE BOMBING ATTACKS DATA STARTED FROM HERE ##################

def Suicide_bombing_attacks(request):
    
    figure_json1 = None
    figure_json2 = None
    figure_json3 = None
    figure_json4 = None
    figure_json5 = None
    figure_json6 = None
    figure_json7 = None
    figure_json8 = None
    figure_json9 = None
    figure_json10 = None
    figure_json11 = None
    figure_json12 = None
    figure_json13 = None
    figure_json14 = None
    figure_json15 = None
    figure_json16 = None
    figure_json17 = None
    figure_json18 = None
    


    import pandas as pd
    import plotly.express as px
    data = StatisticalAnalysis.objects.all()
    for item in data:
        if item.csv_name == 'Suicide_bombing_attacks.csv':
            df = pd.read_csv(item.file.path)
            print(df.head())
            
            # Check if the dataframe has the required columns
            if all(col in df.columns for col in ['Date', 'Islamic Date', 'Blast Day Type', 'Holiday Type', 'Time','City','Latitude', 'Longitude', 'Province', 'Location','Location Category', 'Location Sensitivity', 'Open/Closed Space','Influencing Event/Event', 'Target Type', 'Targeted Sect if any','Killed Min', 'Killed Max', 'Injured Min', 'Injured Max','No. of Suicide Blasts', 'Explosive Weight (max)', 'Hospital Names','Temperature(C)', 'Temperature(F)']):
                    
                
               #####################  On one what day type most of Attacks Happend? starts here ##########################
              

                import plotly.express as px
                import pandas as pd

                # Assuming df is your DataFrame and 'Blast Day Type' is the column of interest
                # Calculate the counts
                counts = df['Blast Day Type'].value_counts().reset_index()
                counts.columns = ['Blast Day Type', 'Count']

                # Create the bar plot
                fig = px.bar(counts, x='Blast Day Type', y='Count')

                # Add annotations (texts on bars)
                for index, row in counts.iterrows():
                    fig.add_annotation(
                        x=row['Blast Day Type'], 
                        y=row['Count'], 
                        text=str(row['Count']),
                        showarrow=False,
                        yshift=10
                    )

                 # Show the plot
                fig1= fig

               #################### On one what day type most of Attacks Happend?  END SESSION ###################



############## Which Province was Attacked most number of Times? start session ###################
               
                # Replace values as per your requirement
                df['Province'].replace({'Fata': 'FATA', 'Baluchistan': 'Balochistan'}, inplace=True)

                # Calculate the counts
                counts = df['Province'].value_counts().reset_index()
                counts.columns = ['Province', 'Count']

                # Create the bar plot
                fig = px.bar(counts, x='Province', y='Count')

                # Add annotations (texts on bars)
                for index, row in counts.iterrows():
                    fig.add_annotation(
                        x=row['Province'], 
                        y=row['Count'], 
                        text=str(row['Count']),
                        showarrow=False,
                        yshift=10
                    )

                fig.update_layout(
                    xaxis_title="Province Name",
                    yaxis_title="Number of Blast Attacks"
                )

                 # Show the plot
                fig2= fig

               
              
               ############# Which Province was Attacked most number of Times? end session ###################



                ######### Top 5 Most Targeted Location Categories starts here ####################

                most_targeted_locations = df['Location Category'].value_counts().head(5)
                # Convert the result to a DataFrame
                most_targeted_locations = pd.DataFrame({'Location Type': most_targeted_locations.index, 'Count': most_targeted_locations.values})
                # Create the pie chart
                fig = px.pie(most_targeted_locations, 
                            values='Count', 
                            names='Location Type', 
                            title='Top 5 Most Targeted Location Categories')

                # Add percentage labels
                fig.update_traces(textinfo='percent+label')

                # Set the start angle
                fig.update_layout(polar=dict(angularaxis=dict(rotation=140)))

                fig3=fig


                ############### Top 5 Most Targeted Location Categories ends here ##################



                ############################# Top 10 Locations of Blast  starts here ###############

                dfLocation = pd.DataFrame(df, columns=['City', 'Province', 'Location', 'No. of Suicide Blasts'])
                dfTopCities = dfLocation.groupby(['City'], as_index=False).sum()

                dfTop10_Cities = dfTopCities.sort_values(["No. of Suicide Blasts"], ascending=False).head(10)
                dfTop10_Cities
                dfTopLocations = dfLocation.groupby(['Location'], as_index=False).sum()
                dfTop10_Locations = dfTopLocations.sort_values(["No. of Suicide Blasts"], ascending=False).head(10)
                # Create the bar plot
                fig = px.bar(dfTop10_Locations, x='Location', y='No. of Suicide Blasts')

                # Update layout
                fig.update_layout(
                    title="Top 10 Locations of Blast",
                    xaxis_title="Location Name",
                    yaxis_title="Number of Suicide Blasts",
                    template="plotly_dark",  # you can change the template as per your preference
                    autosize=False,
                    width=1000,
                    height=800
                )

                # Customize x-axis tick labels rotation
                fig.update_xaxes(tickangle=45)

                fig4=fig

                ######################### Top 10 Locations of Blast  ends here  #######################



                
# Number of Time High, Medium and Low Location where Targeted and in which Province those Location where targeted?starts here  ######

                                
                # Preparing the data
                count_data = df.groupby(['Location Sensitivity', 'Province']).size().reset_index(name='counts')

                # Create the bar plot
                fig = px.bar(count_data, 
                            x='Location Sensitivity', 
                            y='counts', 
                            color='Province', 
                            barmode='group')

                # Update layout
                fig.update_layout(
                    title="Number of Bomb Blasts by Location Sensitivity and Province",
                    xaxis_title="Bomb Blast Location Sensitivity",
                    yaxis_title="Number of Bomb Blasts",
                    template="plotly_dark",  # you can change the template as per your preference
                    autosize=False,
                    width=1000,
                    height=500
                )

                fig5=fig

## Number of Time High, Medium and Low Location where Targeted and in which Province those Location where targeted?ends here  #####


############################## Number of Bomb Blasts by Location Sensitivity and Province starts here ################

             

################################  Number of Bomb Blasts by Location Sensitivity and Province ends here ###################


####################################  Location Sensitivity Numbers ( bar chart ) starts here  #################################### 

                import plotly.express as px
                import pandas as pd

                # Calculate the counts
                counts = df['Location Sensitivity'].value_counts().reset_index()
                counts.columns = ['Location Sensitivity', 'Count']

                # Create the bar plot
                fig = px.bar(counts, x='Location Sensitivity', y='Count')

                # Add annotations (texts on bars)
                for index, row in counts.iterrows():
                    fig.add_annotation(
                        x=row['Location Sensitivity'], 
                        y=row['Count'], 
                        text=str(row['Count']),
                        showarrow=False,
                        yshift=10
                    )

                fig.update_layout(
                    xaxis_title="Location Sensitivity",
                    yaxis_title="Number of Blast Attacks",
                    autosize=False,
                    width=800,
                    height=500
                )

                fig7=fig


        ############### Location Sensitivity Numbers ( bar chart )ends here   #################


            ################################ Top 10 Bomb Blast Target Types starts here ########################
                
                most_targetted_parts = df['Target Type'].value_counts().head(10)
                most_targetted_parts = pd.DataFrame({'Target Type': most_targetted_parts.index, 'Count': most_targetted_parts.values})
                import plotly.express as px
                import pandas as pd


                # Create the bar plot
                fig = px.bar(most_targetted_parts, x='Target Type', y='Count', color='Count')

                # Update layout
                fig.update_layout(
                    title="Top 10 Bomb Blast Target Types",
                    xaxis_title="Bomb Blast Target Type",
                    yaxis_title="Number of Times Specific Target was Considered",
                    template="plotly_dark",  # you can change the template as per your preference
                    autosize=False,
                    width=1000,
                    height=600
                )

                fig8=fig

            ################################ Top 10 Bomb Blast Target Types ends here ########################
            


            ################################ Top 3 Locations with Maximum Killings starts here #################################

                import plotly.express as px
                import pandas as pd

                # Sort the dataframe and select top 7 rows
                df_sorted_killedmax = df.sort_values(by='Killed Max', ascending=False).head(5)

                # Create the bar plot
                fig = px.bar(df_sorted_killedmax, 
                            x='Location', 
                            y='Killed Max', 
                            color='City',
                            barmode='group')

                # Update layout
                fig.update_layout(
                    title="Top 7 Locations with Maximum Killings",
                    xaxis_title="Location",
                    yaxis_title="Number of People Killed",
                    template="plotly_dark",  # you can change the template as per your preference
                    autosize=False,
                    width=1000,
                    height=800
                )

                fig9=fig

            ################################ Top 3 Locations with Maximum Killings ends here #################################


    #####Top 10 Cities with the Highest Number of People Killed by Suicide Bomb Attacks starts here  ##################

                df['Killed Max']=df['Killed Max'].astype('int')
                number_of_people_killed_in_every_city = df.groupby('City', as_index=False)['Killed Max'].sum()
                #now lets sort that dataset to get cities where most number of people were killed in total attacks
                number_of_people_killed_in_every_city.sort_values(by= 'Killed Max', 
                                                                inplace=True,
                                                                ascending=False)

                top_10 = number_of_people_killed_in_every_city.head(10)

                # Create the bar plot
                fig = px.bar(top_10, x='City', y='Killed Max')

                # Update layout
                fig.update_layout(
                    title="Top 10 Cities with the Highest Number of People Killed by Suicide Bomb Attacks",
                    xaxis_title="City",
                    yaxis_title="Total Number of People Killed",
                    template="plotly_dark",  # you can change the template as per your preference
                    autosize=False,
                    width=1000,
                    height=500
                )

                fig10=fig

### Top 10 Cities with the Highest Number of People Killed by Suicide Bomb Attacks ends here #########



#####################   Total Number of Blasts per Year ( bar chart ) starts here    ##########################

                #creating Year Column
                df['Year'] = df['Date'].str.split('-').str[-1]
                Total_Blast_in_every_year = df['Year'].value_counts()
                Total_Blast_in_every_year = pd.DataFrame({'Year': Total_Blast_in_every_year.index, 'Count': Total_Blast_in_every_year.values})

                # Create the bar plot
                fig = px.bar(Total_Blast_in_every_year, x='Year', y='Count')

                # Update layout
                fig.update_layout(
                    title="Total Number of Blasts per Year",
                    xaxis_title="Year",
                    yaxis_title="Total Number of Blasts",
                    template="plotly_dark",  # you can change the template as per your preference
                    autosize=False,
                    width=1000,
                    height=500
                )



                fig11=fig


######################### Total Number of Blasts per Year ( bar chart )  ends here #######################


############################  Total Number of Blasts Every Year ( line chart )   starts here  #############################
                
                sorted_Year = Total_Blast_in_every_year.sort_values(by= 'Year')
                # Create the line plot
                fig = px.line(sorted_Year, x='Year', y='Count')

                # Update layout
                fig.update_layout(
                    title="Total Number of Blasts Every Year",
                    xaxis_title="Year",
                    yaxis_title="Total Number of Blasts",
                    template="plotly_dark",  # you can change the template as per your preference
                    autosize=False,
                    width=1000,
                    height=500,
                    showlegend=False
                )

                # Add grid lines
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=True)

                fig12=fig

############################   Total Number of Blasts Every Year ( line chart )  ends here  #########################



            ####################### Number of Blasts on Specific Sects  starts here ##########################

                        # Handling null values and counting occurrences
                df['Targeted Sect if any'].fillna('No Sect Targeted', inplace=True)
                sect_counts = df['Targeted Sect if any'].value_counts().reset_index()
                sect_counts.columns = ['Targeted Sect', 'Count']

                # Create the bar plot
                fig = px.bar(sect_counts, x='Targeted Sect', y='Count', color='Count')

                # Update layout
                fig.update_layout(
                    title="Number of Blasts on Specific Sects",
                    xaxis_title="Targeted Sect",
                    yaxis_title="Number of Blasts",
                    template="plotly_dark",  # you can change the template as per your preference
                    autosize=False,
                    width=1000,
                    height=500
                )

                fig13=fig

            ############################  Number of Blasts on Specific Sects  ends here ###########################


#################################  Average Number of People Killed and Injured per Year  starts here       ########################

                import plotly.graph_objects as go

                df['Injured Max'] = pd.to_numeric(df['Injured Max'], errors='coerce')
              
                #Replacing Null values in Injured Max Column with mean value
                df['Injured Max'].fillna(df['Injured Max'].mean(),inplace=True)

                #Replacing Null values in Injured Min Column with mean value
                df['Injured Min'].fillna(df['Injured Min'].mean(),inplace=True)

                #Replacing Null values in Killed Min Column with mean value
                df['Killed Min'].fillna(df['Killed Min'].mean(),inplace=True)

                temp_df = df[['Year', 'Killed Max', 'Killed Min', 'Injured Max', 'Injured Min']]


                #finding average number of people killed
                temp_df['people_killed'] = (temp_df['Killed Max'] + temp_df['Killed Min']) / 2

                #finding avaerage number of pople Injured
                temp_df['people_injured'] = (temp_df['Injured Max'] + temp_df['Injured Min']) / 2

                yearly = temp_df.groupby(["Year"], as_index=False).sum()

                # Create the figure object
                fig = go.Figure()

                # Add the 'people_killed' bar
                fig.add_trace(go.Bar(
                    x=yearly['Year'],
                    y=yearly['people_killed'],
                    name='People Killed',
                    marker_color='indianred'
                ))

                # Add the 'people_injured' bar
                fig.add_trace(go.Bar(
                    x=yearly['Year'],
                    y=yearly['people_injured'],
                    name='People Injured',
                    marker_color='lightblue'
                ))

                # Update layout
                fig.update_layout(
                    barmode='group',
                    title="Average Number of People Killed and Injured per Year",
                    xaxis_tickfont_size=14,
                    yaxis=dict(
                        title='Average Number of People',
                        titlefont_size=16,
                        tickfont_size=14,
                    ),
                    legend=dict(
                        x=0,
                        y=1.0,
                        bgcolor='rgba(255, 255, 255, 0)',
                        bordercolor='rgba(255, 255, 255, 0)'
                    ),
                    autosize=False,
                    width=1000,
                    height=500
                )

                fig14=fig

#################################  Average Number of People Killed and Injured per Year ends here     ##########################



###############    Total Number of Victims by Blast Day Type starts here        ##############


                df['Total_Victims'] = temp_df['people_killed'] + temp_df['people_injured']
                df_day_Type_total_victimes= df.groupby('Blast Day Type', as_index=False)['Total_Victims'].sum()
                # Create the donut chart
                fig = px.pie(df_day_Type_total_victimes, 
                            values='Total_Victims', 
                            names='Blast Day Type', 
                            
                            color_discrete_sequence=px.colors.sequential.RdBu)

                # Update layout
                fig.update_layout(
                    title="Total Number of Victims by Blast Day Type",
                    annotations=[dict(text='Blast Day Type', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )

                # Optionally, update the traces to adjust text information
                fig.update_traces(textinfo='percent+label')

                fig15=fig

#################################     Total Number of Victims by Blast Day Type ends here       ####################


###########    A Pie Chart Representing Choice of Terrorists for Bombing  starts here     ###########################

                df['Open/Closed Space'].replace(('open', 'closed', 'Open/Closed', 'Open'), ('Open', 'Closed', 'Open', 'Open'), inplace=True)

                # Calculate the counts
                counts = df['Open/Closed Space'].value_counts().reset_index()
                counts.columns = ['Open/Closed Space', 'Count']

                # Create the pie chart
                fig = px.pie(counts, 
                            values='Count', 
                            names='Open/Closed Space', 
                            title='A Pie Chart Representing Choice of Terrorists for Bombing',
                            color_discrete_sequence=['lightgreen', 'orange'])

                # Explode the 'Closed' slice if needed
                fig.update_traces(textinfo='percent+label', pull=[0, 0.1])

                # Update layout
                fig.update_layout(showlegend=True, legend_title_text='Open/Closed Space')
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))

                fig16=fig

############ A Pie Chart Representing Choice of Terrorists for Bombing   ends here      #######################



#################################     chart 1 starts here      #####################################

                # import folium
                # from folium import plugins
                # from folium.plugins import FastMarkerCluster
                # df.dropna(subset = ['Latitude','Longitude'],inplace=True)
                # map_heatmap = folium.Map([32, 70],tiles = 'CartoDB dark_matter',zoom_start=6)  
                # df_copy = df.copy()
                # df_copy['count'] = 1
                # plugins.HeatMap(data=df_copy[['Latitude', 'Longitude', 'count']].groupby(['Latitude', 'Longitude']).sum().reset_index().values.tolist(), radius=10, max_zoom=13).add_to(map_heatmap)
                # fig17=map_heatmap

#################################     chart 1 ends here      #####################################


#################################      chart 2 starts here     #####################################

                # incidents = folium.Map(location = [27,75], height=1000, width=1000,zoom_start=6)

                # FastMarkerCluster(df[["Latitude", "Longitude"]].values).add_to(incidents)

                # fig18=incidents

#################################    chart 2 ends here       #####################################


                # Convert the figure to JSON format using Plotly's built-in method
                figure_json1 = fig1.to_json()
                figure_json2 = fig2.to_json()
                figure_json3 = fig3.to_json()
                figure_json4 = fig4.to_json()
                figure_json5 = fig5.to_json()
                # figure_json6 = fig6.to_json()
                figure_json7 = fig7.to_json()
                figure_json8 = fig8.to_json()
                figure_json9 = fig9.to_json()
                figure_json10 = fig10.to_json()
                figure_json11 = fig11.to_json()
                figure_json12 = fig12.to_json()
                figure_json13 = fig13.to_json()
                figure_json14 = fig14.to_json()
                figure_json15 = fig15.to_json()
                figure_json16 = fig16.to_json()
                # figure_json17 = fig17.to_json()
                # figure_json18 = fig18.to_json()
                
               


                
            else:
                return HttpResponse("The uploaded CSV is missing some required columns.")

    return render(request, 'Suicide_bombing_attacks.html', {'figure1': figure_json1
    ,'figure2': figure_json2,'figure3': figure_json3,'figure4': figure_json4,'figure5':figure_json5,'figure7':figure_json7,'figure8':figure_json8,'figure9':figure_json9,'figure10':figure_json10,'figure11':figure_json11,'figure12':figure_json12,'figure13':figure_json13,'figure14':figure_json14,'figure15':figure_json15,'figure16':figure_json16,#'figure17':figure_json17,'figure18':figure_json18
    })

            

#################################### PAKISTAN SUICIDE BOMBINGS DATA ENDS HERE ###############################






###***********************************************

def statistics_page(request):
   
    fig_all_cities_on_map_json = None
    fig_most_badly_effected_district= None
    fig_most_badly_effected_division = None
    fig_crimes_category_ranking = None
    fig_crime_type_division_year_analysis = None
    fig_crimetype_years_trend = None
    fig_divisions_years_trends = None
    fig_crimes_years_trends = None

    fig_001_map_json = None
    fig_002_map_json = None
    fig_003_map_json = None
    fig_004_map_json = None
    fig_005_map_json = None
    fig_006_map_json = None
    fig_007_map_json = None
    fig_008_map_json = None
    fig_009_map_json = None
    
    fig001_json = None
    fig002_json = None
    fig003_json = None
    fig004_json = None
    fig005_json = None
    fig006_json = None
    fig007_json = None
    fig008_json = None
    fig009_json = None
   
    fig1_json = None
    fig2_json = None
    fig3_json = None
    fig4_json = None
    fig5_json = None
    fig6_json = None
    fig7_json = None

    file_summary = ""
    analysis1 = ""
    analysis2 = ""
    analysis3 = ""
    analysis4 = ""

    data = StatisticalAnalysis.objects.all()
    for item in data:
        if item.csv_name == 'pak_crime_stats.csv':
            df = pd.read_csv(item.file.path)

            # Check if the dataframe has the required columns
            if all(col in df.columns for col in ['Year', 'Division', 'District', 'CrimeType', 'CrimeCount', 'Population']):
                
                # File Summary
                total_years = df['Year'].nunique()
                total_districts = df['District'].nunique()
                total_crimes = df['CrimeCount'].sum()
                avg_crimes_per_year = int(df.groupby('Year')['CrimeCount'].sum().mean())
                file_summary = f"This data covers a span of {total_years} years across {total_districts} districts. A total of {total_crimes} crimes were reported, with an average of {avg_crimes_per_year} crimes per year."
                
                # 1. Yearly Crime Trend
                # yearly_crime_trend_data = df.groupby('Year')['CrimeCount'].sum().reset_index()
                # fig_yearly_crime_trend = go.Figure(data=go.Scatter(x=yearly_crime_trend_data['Year'], 
                #                                                     y=yearly_crime_trend_data['CrimeCount'],
                #                                                     mode='lines+markers+text',
                #                                                     text=yearly_crime_trend_data['CrimeCount'],
                #                                                     textposition='top center'))
                # fig_yearly_crime_trend.update_layout(title='Yearly Crime Trend',
                #                                      xaxis_title='Year',
                #                                      yaxis_title='Number of Crimes',
                #                                      xaxis=dict(showgrid=True, gridwidth=0.5),
                #                                      yaxis=dict(showgrid=True, gridwidth=0.5))
                # max_year_crime = yearly_crime_trend_data['CrimeCount'].max()
                # year_max_crime = yearly_crime_trend_data[yearly_crime_trend_data['CrimeCount'] == max_year_crime]['Year'].values[0]
                # analysis1 = f"The year with the highest number of reported crimes was {year_max_crime} with a total of {max_year_crime} crimes."

                


                #############  crime analysis of each divions with respect to their districts starts here ##############

                ## 0.1 Crime trends in Bahawalpur

                
                # Filter the data for the 'Bahawalpur' division and create a copy
                bahawalpur_data = df[df['Division'] == 'Bahawalpur'].copy()

                # Convert the 'Year' column to a string
                bahawalpur_data['Year'] = bahawalpur_data['Year'].astype(str)

                # Generate the segmented bar plot
                fig001 = px.bar(bahawalpur_data, 
                            x="District", 
                            y="CrimeCount", 
                            color="Year",  # Segment bars by Year
                           
                            hover_data={"CrimeCount": ":,"},  # Format hover data as a simple number
                            template='plotly_dark'
                            )

                ## map of bahawalpur division and its districts

                import plotly.graph_objects as go

                # Filter the data for the 'Bahawalpur' division and create a copy
                bahawalpur_data = df[df['Division'] == 'Bahawalpur'].copy()

                # Aggregate data by District to get total CrimeCount
                agg_bahawalpur_data = bahawalpur_data.groupby('District')['CrimeCount'].sum().reset_index()

                # Coordinates for districts in the Bahawalpur division and the division itself
                district_coords_updated = {
                    "Bahawalpur": (28.5062, 71.5724),  # Division center
                    "R.Y.Khan": (28.5494, 70.6217),
                    "Bahawalnagar": (29.6892, 72.9933)
                }

                # Modified code to handle potential missing districts in the aggregated data

                # Creating the scatter mapbox plot centered around Pakistan
                fig = go.Figure()
                available_districts = ['Bahawalpur','R.Y.Khan','Bahawalnagar']
                # Adding data points for Bahawalpur division and its districts
                for location, coord in district_coords_updated.items():
                    if location in available_districts:  # Check if location exists in the data
                        crime_count = agg_bahawalpur_data[agg_bahawalpur_data['District'] == location]['CrimeCount'].values[0]
                        hover_text = f"{location}<br>Total Crime Count: {crime_count}"
                    else:
                        hover_text = location  # Just the division name for the center

                    fig.add_trace(go.Scattermapbox(
                        lon=[coord[1]],
                        lat=[coord[0]],
                        hovertemplate=hover_text,
                        mode='markers',
                        marker=dict(size=20 if location == "Bahawalpur Division" else 10),
                        name=location
                    ))

                # Updating layout for the map to center around Pakistan
                fig_001_map=fig.update_layout(
                     title="Bahawalpur Division and Its Districts",
                    mapbox=dict(
                        center=dict(lat=30.3753, lon=69.3451),
                        zoom=5,
                        style="open-street-map",
                        bearing=0,
                        pitch=0
                    ),
                    autosize=True,

                )

                


                ## 0.2 Crime trends in Bahawalpur with respect to their district

                # Filter the data for the 'Dera Ghazi Khan' division and create a copy
                dera_ghazi_khan_data = df[df['Division'] == 'Dera Ghazi Khan'].copy()

                # Convert the 'Year' column to a string
                dera_ghazi_khan_data['Year'] = dera_ghazi_khan_data['Year'].astype(str)

                # Generate the segmented bar plot
                fig002 = px.bar(dera_ghazi_khan_data, 
                            x="District", 
                            y="CrimeCount", 
                            color="Year",  # Segment bars by Year
                            title="Total CrimeCount in Dera Ghazi Khan Division by District Segmented by Year",
                            hover_data={"CrimeCount": ":,"}  # Format hover data as a simple number
                            )


                # map of dera ghazi khan
                                # Filter the data for the 'Bahawalpur' division and create a copy
                dera_ghazi_khan_data = df[df['Division'] == 'Dera Ghazi Khan'].copy()

                # Aggregate data by District to get total CrimeCount
                agg_dera_ghazi_khan_data = dera_ghazi_khan_data.groupby('District')['CrimeCount'].sum().reset_index()



                # Coordinates for districts in the Bahawalpur division and the division itself
                district_coords_updated = {
                    "D.G.Khan": (30.0489, 70.6455),
                    "Layyah": (30.9693, 70.9428),  # Division center
                    "Muzaffargarh": (30.0736, 71.1805),
                    "Rajanpur": ( 29.1044, 70.3301)
                }

                # Modified code to handle potential missing districts in the aggregated data

                # Creating the scatter mapbox plot centered around Pakistan
                fig = go.Figure()
                available_districts = ['D.G.Khan','Layyah','Muzaffargarh','Rajanpur']
                # Adding data points for Bahawalpur division and its districts
                for location, coord in district_coords_updated.items():
                    if location in available_districts:  # Check if location exists in the data
                        crime_count = agg_dera_ghazi_khan_data[agg_dera_ghazi_khan_data['District'] == location]['CrimeCount'].values[0]
                        hover_text = f"{location}<br>Total Crime Count: {crime_count}"
                    else:
                        hover_text = location  # Just the division name for the center

                    fig.add_trace(go.Scattermapbox(
                        lon=[coord[1]],
                        lat=[coord[0]],
                        hovertemplate=hover_text,
                        mode='markers',
                        marker=dict(size=20 if location == "D.G.Khan Division" else 10),
                        name=location
                    ))

                # Updating layout for the map to center around Pakistan
                fig_002_map=fig.update_layout(
                    title="Dera Ghazi Khan Division and Its Districts",
                    mapbox=dict(
                        center=dict(lat=30.3753, lon=69.3451),
                        zoom=5,
                        style="open-street-map",
                        bearing=0,
                        pitch=0
                    ),
                    autosize=True,
                    plot_bgcolor='black',  # Set the background color to black
                    paper_bgcolor='black',  # Set the paper (plot area) background color to black
                    legend_title_font_color='white',  # Set legend title color to white
                    legend_font_color='white',  # Set legend text color to white
                    title_font_color = 'white'
                )

           

                ## 0.3 Crime trends in Faisalabad with respect to their district

                # Filter the data for the 'Bahawalpur' division and create a copy
                faisalabad_data = df[df['Division'] == 'Faisalabad'].copy()

                # Convert the 'Year' column to a string
                faisalabad_data['Year'] = faisalabad_data['Year'].astype(str)

                # Generate the segmented bar plot
                fig003 = px.bar(faisalabad_data, 
                            x="District", 
                            y="CrimeCount", 
                            color="Year",  # Segment bars by Year
                            title="Total CrimeCount in Faisalabad Division by District Segmented by Year",
                            hover_data={"CrimeCount": ":,"},  # Format hover data as a simple number
                            template='plotly_dark'
                            )

                ## map of faisalabad
                # Filter the data for the 'Bahawalpur' division and create a copy
                faisalabad_data = df[df['Division'] == 'Faisalabad'].copy()

                # Aggregate data by District to get total CrimeCount
                agg_faisalabad_data = faisalabad_data.groupby('District')['CrimeCount'].sum().reset_index()



                # Coordinates for districts in the Bahawalpur division and the division itself
                district_coords_updated = {
                    "Faisalabad": (31.4504, 73.1350),
                    "Jhang": (31.2781, 72.3317),  # Division center
                    "T.T.Singh": ( 30.9709, 72.4826),
                    "Chiniot": ( 31.7292, 72.9822)
                }

                # Modified code to handle potential missing districts in the aggregated data

                # Creating the scatter mapbox plot centered around Pakistan
                fig = go.Figure()
                available_districts = ['Faisalabad','Jhang','T.T.Singh','Chiniot']
                # Adding data points for Bahawalpur division and its districts
                for location, coord in district_coords_updated.items():
                    if location in available_districts:  # Check if location exists in the data
                        crime_count = agg_faisalabad_data[agg_faisalabad_data['District'] == location]['CrimeCount'].values[0]
                        hover_text = f"{location}<br>Total Crime Count: {crime_count}"
                    else:
                        hover_text = location  # Just the division name for the center

                    fig.add_trace(go.Scattermapbox(
                        lon=[coord[1]],
                        lat=[coord[0]],
                        hovertemplate=hover_text,
                        mode='markers',
                        marker=dict(size=20 if location == "Faisalabad Division" else 10),
                        name=location
                    ))

                # Updating layout for the map to center around Pakistan
                fig_003_map=fig.update_layout(
                    title="Faisalabad Division and Its Districts",
                    mapbox=dict(
                        center=dict(lat=30.3753, lon=69.3451),
                        zoom=5,
                        style="open-street-map",
                        bearing=0,
                        pitch=0
                    ),
                    autosize=True
                )

                

                ## 0.4 Crime trends in Gujranwala with respect to their district

                # Filter the data for the 'Gujranwala' division and create a copy

                Gujranwala_data = df[df['Division'] == 'Gujranwala'].copy()

                # Convert the 'Year' column to a string
                Gujranwala_data['Year'] = Gujranwala_data['Year'].astype(str)

                # Generate the segmented bar plot
                fig004 = px.bar(Gujranwala_data, 
                            x="District", 
                            y="CrimeCount", 
                            color="Year",  # Segment bars by Year
                            title="Total CrimeCount in Gujranwala Division by District Segmented by Year",
                            hover_data={"CrimeCount": ":,"}  # Format hover data as a simple number
                            )

                ## map of gujranwala
                # Filter the data for the 'gujranwala' division and create a copy
                gujranwala_data = df[df['Division'] == 'Gujranwala'].copy()

                # Aggregate data by District to get total CrimeCount
                agg_gujranwala_data = gujranwala_data.groupby('District')['CrimeCount'].sum().reset_index()



                # Coordinates for districts in the Bahawalpur division and the division itself
                district_coords_updated = {
                    "Gujranwala": (32.1877, 74.1945),
                    "Gujrat": (32.5731, 74.1005),  # Division center
                    "Hafizabad": ( 32.0712, 73.6895),
                    "Mandi Baha-ud-Din": ( 32.5742, 73.4828),
                    "Narowal": ( 32.1014, 74.8800),
                    "Sialkot": ( 32.4945, 74.5229),


                }

                # Modified code to handle potential missing districts in the aggregated data

                # Creating the scatter mapbox plot centered around Pakistan
                fig = go.Figure()
                available_districts = ['Gujranwala','Gujrat','Hafizabad','Mandi Baha-ud-Din','Narowal','Sialkot']
                # Adding data points for Bahawalpur division and its districts
                for location, coord in district_coords_updated.items():
                    if location in available_districts:  # Check if location exists in the data
                        crime_count = agg_gujranwala_data[agg_gujranwala_data['District'] == location]['CrimeCount'].values[0]
                        hover_text = f"{location}<br>Total Crime Count: {crime_count}"
                    else:
                        hover_text = location  # Just the division name for the center

                    fig.add_trace(go.Scattermapbox(
                        lon=[coord[1]],
                        lat=[coord[0]],
                        hovertemplate=hover_text,
                        mode='markers',
                        marker=dict(size=20 if location == "Gujranwala Division" else 10),
                        name=location
                    ))

                # Updating layout for the map to center around Pakistan
                fig_004_map=fig.update_layout(
                    title="Gujranwala Division and Its Districts",
                    mapbox=dict(
                        center=dict(lat=30.3753, lon=69.3451),
                        zoom=5,
                        style="open-street-map",
                        bearing=0,
                        pitch=0,
                        
                    ),
                    autosize=True,
                    plot_bgcolor='black',  # Set the background color to black
                    paper_bgcolor='black',  # Set the paper (plot area) background color to black
                    legend_title_font_color='white',  # Set legend title color to white
                    legend_font_color='white',  # Set legend text color to white
                    title_font_color = 'white'
                )

              
                ## 0.5 Crime trends in Lahore with respect to their district

                # Filter the data for the 'Bahawalpur' division and create a copy
                Lahore_data = df[df['Division'] == 'Lahore'].copy()

                # Convert the 'Year' column to a string
                Lahore_data['Year'] = Lahore_data['Year'].astype(str)

                # Generate the segmented bar plot
                fig005 = px.bar(Lahore_data, 
                            x="District", 
                            y="CrimeCount", 
                            color="Year",  # Segment bars by Year
                            title="Total CrimeCount in Lahore Division by District Segmented by Year",
                            hover_data={"CrimeCount": ":,"},  # Format hover data as a simple number
                            template='plotly_dark'
                            )
                
                ## map of lahore
                # Filter the data for the 'Lahore' division and create a copy
                Lahore_data = df[df['Division'] == 'Lahore'].copy()

                # Aggregate data by District to get total CrimeCount
                agg_Lahore_data = Lahore_data.groupby('District')['CrimeCount'].sum().reset_index()



                # Coordinates for districts in the Bahawalpur division and the division itself
                district_coords_updated = {
                    "Lahore": ( 31.558, 74.3507),
                    "Kasur": ( 31.118793, 74.463272),  # Division center
                    "Okara": ( 30.808500, 73.459396),
                    "Sheikhupura": ( 31.716661, 73.985023),
                    "Nankana Sahib": ( 31.452097, 73.708305),


                }

                # Modified code to handle potential missing districts in the aggregated data

                # Creating the scatter mapbox plot centered around Pakistan
                fig = go.Figure()
                available_districts = ['Lahore','Kasur','Okara','Sheikhupura','Nankana Sahib']
                # Adding data points for Bahawalpur division and its districts
                for location, coord in district_coords_updated.items():
                    if location in available_districts:  # Check if location exists in the data
                        crime_count = agg_Lahore_data[agg_Lahore_data['District'] == location]['CrimeCount'].values[0]
                        hover_text = f"{location}<br>Total Crime Count: {crime_count}"
                    else:
                        hover_text = location  # Just the division name for the center

                    fig.add_trace(go.Scattermapbox(
                        lon=[coord[1]],
                        lat=[coord[0]],
                        hovertemplate=hover_text,
                        mode='markers',
                        marker=dict(size=20 if location == "Lahore Division" else 10),
                        name=location
                    ))

                # Updating layout for the map to center around Pakistan
                fig_005_map=fig.update_layout(
                    title="Lahore Division and Its Districts",
                    mapbox=dict(
                        center=dict(lat=30.3753, lon=69.3451),
                        zoom=5,
                        style="open-street-map",
                        bearing=0,
                        pitch=0
                    ),
                    autosize=True
                )

                

                ## 0.6 Crime trends in Multan with respect to their district

                # Filter the data for the 'Multan' division and create a copy
                Multan_data = df[df['Division'] == 'Multan'].copy()

                # Convert the 'Year' column to a string
                Multan_data['Year'] = Multan_data['Year'].astype(str)

                # Generate the segmented bar plot
                fig006 = px.bar(Multan_data, 
                            x="District", 
                            y="CrimeCount", 
                            color="Year",  # Segment bars by Year
                            title="Total CrimeCount in Multan Division by District Segmented by Year",
                            hover_data={"CrimeCount": ":,"},  # Format hover data as a simple number
                            template='plotly_dark'
                            )

                # map of multan
                # Filter the data for the 'Multan' division and create a copy
                Multan_data = df[df['Division'] == 'Multan'].copy()

                # Aggregate data by District to get total CrimeCount
                agg_Multan_data = Multan_data.groupby('District')['CrimeCount'].sum().reset_index()



                # Coordinates for districts in the Multan division and the division itself
                district_coords_updated = {
                    "Multan": ( 30.1968, 71.4782),
                    "Khanewal": ( 30.286415, 71.932030),  # Division center
                    "Lodhran": ( 29.5405100, 71.6335700),
                    "Vehari": ( 30.0333300, 72.3500000),
                    "Sahiwal": ( 30.6611813, 73.1085756),
                    "Pakpattan": ( 30.00000000, 70.00000000),
                }

                # Modified code to handle potential missing districts in the aggregated data

                # Creating the scatter mapbox plot centered around Pakistan
                fig = go.Figure()
                available_districts = ['Multan','Khanewal','Lodhran','Sahiwal','Pakpattan']
                # Adding data points for Multan division and its districts
                for location, coord in district_coords_updated.items():
                    if location in available_districts:  # Check if location exists in the data
                        crime_count = agg_Multan_data[agg_Multan_data['District'] == location]['CrimeCount'].values[0]
                        hover_text = f"{location}<br>Total Crime Count: {crime_count}"
                    else:
                        hover_text = location  # Just the division name for the center

                    fig.add_trace(go.Scattermapbox(
                        lon=[coord[1]],
                        lat=[coord[0]],
                        hovertemplate=hover_text,
                        mode='markers',
                        marker=dict(size=20 if location == "Multan Division" else 10),
                        name=location
                    ))

                # Updating layout for the map to center around Pakistan
                fig_006_map=fig.update_layout(
                    title="Multan Division and Its Districts",
                    mapbox=dict(
                        center=dict(lat=30.3753, lon=69.3451),
                        zoom=5,
                        style="open-street-map",
                        bearing=0,
                        pitch=0
                    ),
                    autosize=True,
                  
                )

                
                ## 0.7 Crime trends in Rawalpindi with respect to their district

                # Filter the data for the 'Rawalpindi' division and create a copy
                Rawalpindi_data = df[df['Division'] == 'Rawalpindi'].copy()

                # Convert the 'Year' column to a string
                Rawalpindi_data['Year'] = Rawalpindi_data['Year'].astype(str)

                # Generate the segmented bar plot
                fig007 = px.bar(Rawalpindi_data, 
                            x="District", 
                            y="CrimeCount", 
                            color="Year",  # Segment bars by Year
                            title="Total CrimeCount in Lahore Division by District Segmented by Year",
                            hover_data={"CrimeCount": ":,"},  # Format hover data as a simple number
                            template='ggplot2'
                            )

                
                # map of rawalpindi
                # Filter the data for the 'Rawalpindi' division and create a copy
                Rawalpindi_data = df[df['Division'] == 'Rawalpindi'].copy()

                # Aggregate data by District to get total CrimeCount
                agg_Rawalpindi_data = Rawalpindi_data.groupby('District')['CrimeCount'].sum().reset_index()



                # Coordinates for districts in the Rawalpindi division and the division itself
                district_coords_updated = {
                    "Rawalpindi": ( 33.5651, 73.0169),
                    "Attock": ( 33.7660, 72.3609),  # Division center
                    "Chakwal": ( 32.9328, 72.8630),
                    "Jhelum": ( 32.9425, 73.7257),
                }

                # Modified code to handle potential missing districts in the aggregated data

                # Creating the scatter mapbox plot centered around Pakistan
                fig = go.Figure()
                available_districts = ['Rawalpindi','Attock','Chakwal','Jhelum']
                # Adding data points for Multan division and its districts
                for location, coord in district_coords_updated.items():
                    if location in available_districts:  # Check if location exists in the data
                        crime_count = agg_Rawalpindi_data[agg_Rawalpindi_data['District'] == location]['CrimeCount'].values[0]
                        hover_text = f"{location}<br>Total Crime Count: {crime_count}"
                    else:
                        hover_text = location  # Just the division name for the center

                    fig.add_trace(go.Scattermapbox(
                        lon=[coord[1]],
                        lat=[coord[0]],
                        hovertemplate=hover_text,
                        mode='markers',
                        marker=dict(size=20 if location == "Rawalpindi Division" else 10),
                        name=location
                    ))

                # Updating layout for the map to center around Pakistan
                fig_007_map=fig.update_layout(
                    title="Rawalpindi Division and Its Districts",
                    mapbox=dict(
                        center=dict(lat=30.3753, lon=69.3451),
                        zoom=5,
                        style="open-street-map",
                        bearing=0,
                        pitch=0
                    ),
                    autosize=True,
                    plot_bgcolor='black',  # Set the background color to black
                    paper_bgcolor='black',  # Set the paper (plot area) background color to black
                    legend_title_font_color='white',  # Set legend title color to white
                    legend_font_color='white',  # Set legend text color to white
                    title_font_color = 'white'
                )

                
                ## 0.8 Crime trends in Sargodha with respect to their district

                # Filter the data for the 'Sargodha' division and create a copy
                Sargodha_data = df[df['Division'] == 'Sargodha'].copy()

                # Convert the 'Year' column to a string
                Sargodha_data['Year'] = Sargodha_data['Year'].astype(str)

                # Generate the segmented bar plot
                fig008 = px.bar(Sargodha_data, 
                            x="District", 
                            y="CrimeCount", 
                            color="Year",  # Segment bars by Year
                            title="Total CrimeCount in Sargodha Division by District Segmented by Year",
                            hover_data={"CrimeCount": ":,"},  # Format hover data as a simple number
                            template='plotly_dark'
                            )

                ## map of sargodha
                # Filter the data for the 'Sargodha' division and create a copy
                Sargodha_data = df[df['Division'] == 'Sargodha'].copy()

                # Aggregate data by District to get total CrimeCount
                agg_Sargodha_data = Sargodha_data.groupby('District')['CrimeCount'].sum().reset_index()



                # Coordinates for districts in the Sargodha division and the division itself
                district_coords_updated = {
                    "Sargodha": ( 32.0740, 72.6861),
                    "Bhakkar": ( 31.6082, 71.0854),  # Division center
                    "Khushab": ( 32.2955, 72.3489),
                    "Mianwali": ( 32.5839, 71.5370),
                }

                # Modified code to handle potential missing districts in the aggregated data

                # Creating the scatter mapbox plot centered around Pakistan
                fig = go.Figure()
                available_districts = ['Sargodha','Bhakkar','Khushab','Mianwali']
                # Adding data points for Sargodha division and its districts
                for location, coord in district_coords_updated.items():
                    if location in available_districts:  # Check if location exists in the data
                        crime_count = agg_Sargodha_data[agg_Sargodha_data['District'] == location]['CrimeCount'].values[0]
                        hover_text = f"{location}<br>Total Crime Count: {crime_count}"
                    else:
                        hover_text = location  # Just the division name for the center

                    fig.add_trace(go.Scattermapbox(
                        lon=[coord[1]],
                        lat=[coord[0]],
                        hovertemplate=hover_text,
                        mode='markers',
                        marker=dict(size=20 if location == "Sargodha Division" else 10),
                        name=location
                    ))

                # Updating layout for the map to center around Pakistan
                fig_008_map=fig.update_layout(
                    title="Sargodha Division and Its Districts",
                    mapbox=dict(
                        center=dict(lat=30.3753, lon=69.3451),
                        zoom=5,
                        style="open-street-map",
                        bearing=0,
                        pitch=0
                    ),
                    autosize=True,
                    
                )

                
                ## 0.9 Crime trends in Sahiwal with respect to their district
                
                # Filter the data for the 'Sahiwal' division and create a copy
                Sahiwal_data = df[df['Division'] == 'Sahiwal'].copy()

                # Convert the 'Year' column to a string
                Sahiwal_data['Year'] = Sahiwal_data['Year'].astype(str)

                # Generate the segmented bar plot
                fig009 = px.bar(Sahiwal_data, 
                            x="District", 
                            y="CrimeCount", 
                            color="Year",  # Segment bars by Year
                            title="Total CrimeCount in Sahiwal Division by District Segmented by Year",
                            hover_data={"CrimeCount": ":,"},  # Format hover data as a simple number
                            
                            )

                ## map of sahiwal
                                # Filter the data for the 'Sahiwal' division and create a copy
                Sahiwal_data = df[df['Division'] == 'Sahiwal'].copy()

                # Aggregate data by District to get total CrimeCount
                agg_Sahiwal_data = Sahiwal_data.groupby('District')['CrimeCount'].sum().reset_index()



                # Coordinates for districts in the Sahiwal division and the division itself
                district_coords_updated = {
                    "Sahiwal": ( 30.6682, 73.1114),
                    "Okara": ( 30.8138, 73.4534),  # Division center
                    "Pakpattan": ( 30.3495, 73.3827),
                }

                # Modified code to handle potential missing districts in the aggregated data

                # Creating the scatter mapbox plot centered around Pakistan
                fig = go.Figure()
                available_districts = ['Sahiwal','Okara','Pakpattan']
                # Adding data points for Sahiwal division and its districts
                for location, coord in district_coords_updated.items():
                    if location in available_districts:  # Check if location exists in the data
                        crime_count = agg_Sahiwal_data[agg_Sahiwal_data['District'] == location]['CrimeCount'].values[0]
                        hover_text = f"{location}<br>Total Crime Count: {crime_count}"
                    else:
                        hover_text = location  # Just the division name for the center

                    fig.add_trace(go.Scattermapbox(
                        lon=[coord[1]],
                        lat=[coord[0]],
                        hovertemplate=hover_text,
                        mode='markers',
                        marker=dict(size=20 if location == "Sahiwal Division" else 10),
                        name=location
                    ))

                # Updating layout for the map to center around Pakistan
                fig_009_map=fig.update_layout(
                    title="Sahiwal Division and Its Districts",
                    mapbox=dict(
                        center=dict(lat=30.3753, lon=69.3451),
                        zoom=5,
                        style="open-street-map",
                        bearing=0,
                        pitch=0
                    ),
                    autosize=True,
                    plot_bgcolor='black',  # Set the background color to black
                    paper_bgcolor='black',  # Set the paper (plot area) background color to black
                    legend_title_font_color='white',  # Set legend title color to white
                    legend_font_color='white',  # Set legend text color to white
                    title_font_color = 'white'
                )
                
                            
   #############  crime analysis of each divions with respect to their districts ends here ##############

                # 1. Yearly Crime Trend Bubble Chart
                yearly_crime_trend_data = df.groupby('Year')['CrimeCount'].sum().reset_index()

                # Create a bubble chart
                fig_yearly_crime_trend = go.Figure(data=go.Scatter(x=yearly_crime_trend_data['Year'], 
                                                                y=yearly_crime_trend_data['CrimeCount'],
                                                                mode='markers+text',
                                                                text=yearly_crime_trend_data['CrimeCount'],
                                                    
                                                                textposition='top center',
                                                                # marker=dict(size=yearly_crime_trend_data['CrimeCount'] / max(yearly_crime_trend_data['CrimeCount']) * 50) # Adjust the size
                                                                marker=dict(size=yearly_crime_trend_data['CrimeCount'] / max(yearly_crime_trend_data['CrimeCount']) * 50, # Adjust the size
                                                                color=yearly_crime_trend_data['Year'], # Use 'Year' as color scale
                                                                colorscale='Viridis', # Color scale
                                                                sizemode='diameter', # Marker size by diameter
                                                                colorbar=dict(title='Year') # Color bar title
                                                              )
                                                                ))
                fig_yearly_crime_trend.update_layout(title='Yearly Crime Trend',
                                                    xaxis_title='Year',
                                                    yaxis_title='Number of Crimes',
                        
                                                    xaxis=dict(showgrid=True, gridwidth=0.5),
                                                    yaxis=dict(showgrid=True, gridwidth=0.5))
                max_year_crime = yearly_crime_trend_data['CrimeCount'].max()
                year_max_crime = yearly_crime_trend_data[yearly_crime_trend_data['CrimeCount'] == max_year_crime]['Year'].values[0]
                analysis1 = f"The year with the highest number of reported crimes was {year_max_crime} with a total of {max_year_crime} crimes."





                # 2. Most Common Crimes
                # crime_counts = df.groupby('CrimeType')['CrimeCount'].sum().sort_values(ascending=False).head(10)
                # fig_most_common_crimes = px.bar(crime_counts.reset_index(),
                #                                 x='CrimeType', y='CrimeCount',
                #                                 title='Most Common Crimes',
                #                                  color='CrimeType')
                # top_crime = crime_counts.idxmax()
                # top_crime_count = crime_counts.max()
                # analysis2 = f"The most common crime is '{top_crime}' with {top_crime_count} reported cases."




                ########################### all of the cities now on the map section starts ########################

                import plotly.graph_objects as go

                # Coordinates for districts in Pakistan
                district_coords = {
                    "Bahawalpur": (28.5062, 71.5724),
                    "R.Y.Khan": (28.5494, 70.6217),
                    "Bahawalnagar": (29.6892, 72.9933),
                    "D.G.Khan": (30.0489, 70.6455),
                    "Layyah": (30.9693, 70.9428),
                    "Muzaffargarh": (30.0736, 71.1805),
                    "Rajanpur": (29.1044, 70.3301),
                    "Faisalabad": (31.4504, 73.1350),
                    "Jhang": (31.2781, 72.3317),
                    "T.T.Singh": (30.9709, 72.4826),
                    "Chiniot": (31.7292, 72.9822),
                    "Gujranwala": (32.1877, 74.1945),
                    "Gujrat": (32.5731, 74.1005),
                    "Hafizabad": (32.0712, 73.6895),
                    "Mandi Baha-ud-Din": (32.5742, 73.4828),
                    "Narowal": (32.1014, 74.8800),
                    "Sialkot": (32.4945, 74.5229),
                    "Lahore": (31.558, 74.3507),
                    "Kasur": (31.118793, 74.463272),
                    "Okara": (30.808500, 73.459396),
                    "Sheikhupura": (31.716661, 73.985023),
                    "Nankana Sahib": (31.452097, 73.708305),
                    "Multan": (30.1968, 71.4782),
                    "Khanewal": (30.286415, 71.932030),
                    "Lodhran": (29.5405100, 71.6335700),
                    "Vehari": (30.0333300, 72.3500000),
                    "Sahiwal": (30.6682, 73.1114),
                    "Pakpattan": (30.3495, 73.3827),
                    "Rawalpindi": (33.5651, 73.0169),
                    "Attock": (33.7660, 72.3609),
                    "Chakwal": (32.9328, 72.8630),
                    "Jhelum": (32.9425, 73.7257),
                    "Sargodha": (32.0740, 72.6861),
                    "Bhakkar": (31.6082, 71.0854),
                    "Khushab": (32.2955, 72.3489),
                    "Mianwali": (32.5839, 71.5370),
                }

                # Create a scatter mapbox plot centered around Pakistan
                fig = go.Figure()

                # Adding data points for each district
                for location, coord in district_coords.items():
                    fig.add_trace(go.Scattermapbox(
                        lon=[coord[1]],
                        lat=[coord[0]],
                        mode='markers',
                        marker=dict(size=10),
                        text=location,
                        name=location
                    ))

                # Updating layout for the map to center around Pakistan
                all_cities_on_map = fig.update_layout(
                    mapbox=dict(
                        center=dict(lat=30.3753, lon=69.3451),
                        zoom=5,
                        style="open-street-map",
                        
                    ),
                    autosize=True,
                    showlegend=False
                )


####################### all of the cities now on the map section end #################################


###################### start - which district has been effected badly , the analysis of which district got more casulaites ##########

                # Aggregate data
                agg_data = df.groupby('District')['CrimeCount'].sum().reset_index()

                # Create a horizontal bar plot
                fig = px.bar(
                    agg_data,
                    x='CrimeCount',
                    y='District',
                    orientation='h',
                    color='District',
                    title="Total Crime Count by District",
                    hover_data={'CrimeCount': ':,'}  # Format hover data as a simple number

                )

                
                most_badly_effected_district = fig.update_layout(
                     template='plotly_dark',
                    plot_bgcolor='#23272c',
                    paper_bgcolor='#23272c',
                    title_font=dict(size=24, color='#FFFFFF'),
                    font=dict(color='#FFFFFF'),
                    xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='#FFFFFF'),
                    yaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='#FFFFFF'),

                    showlegend=False)

               

###################### end -which district has been effected badly , the analysis of which district got more casulaites ###########


                ###### start - which division badly impacted ###############

                # Aggregate data pie chart
                agg_data = df.groupby('Division')['CrimeCount'].sum().reset_index()

                # Create a pie chart
                fig = px.pie(
                    agg_data,
                    values='CrimeCount',
                    names='Division',
                    title="Percentage of Total Crime Count by Division",
                    hover_data={'CrimeCount': ':,'},  # Format hover data as a simple number
                    hole=0.3  # Size of the hole in the middle
                )

                most_badly_effected_division = fig.update_layout(showlegend=True)
                

                ########### end - which divions badly impacted ###################


                ########## ranking of crimes over the whole period of data start #####################

                # Aggregate data for funnel chart
                agg_data = df.groupby('CrimeType')['CrimeCount'].sum().reset_index()

                # Sort the data by CrimeCount in descending order
                agg_data_sorted = agg_data.sort_values(by='CrimeCount', ascending=True)

                # Create a funnel chart
                fig = px.funnel(
                    agg_data_sorted,
                    x='CrimeCount',
                    y='CrimeType',
                    color='CrimeType',
                    template = 'plotly_dark',
                    title="Total Crime Count by Crime Type",
                    hover_data={'CrimeCount': ':,'}  # Format hover data as a simple number
                )

                # Update layout to adjust the size and add axis labels
                crimes_category_ranking = fig.update_layout(
                    xaxis_title="Total Crime Count",
                    yaxis_title="Crime Type",
                    showlegend=False
                )

                

                ############### ranking of crimes over the whole period of data ends ################


                #######################################  crimetype , divison , year analaysis starts here #####################

                
                # Define the CrimeTypes to exclude
                exclude_crime_types = ['Assault on Govt. Servants', 'Rioting', 'Rape', 'Misc.']

                # Filter out unwanted CrimeTypes
                data_filtered = df[~df['CrimeType'].isin(exclude_crime_types)]

                # Aggregate data by Year, Division, and CrimeType
                agg_data = data_filtered.groupby(['Year', 'Division', 'CrimeType'])['CrimeCount'].sum().reset_index()

                # Create a bar plot
                fig = px.bar(
                    agg_data,
                    x='Year',
                    y='CrimeCount',
                    color='Division',
                    title="Total Crime Count of Each Crime Category by Year and Division",
                    hover_data={'CrimeCount': ':,'},  # Format hover data as a simple number
                    facet_col='CrimeType',  # Create separate plots for each CrimeType
                    facet_col_wrap=2,
                    category_orders={'CrimeType': agg_data['CrimeType'].unique()}  # Order facets by CrimeType
                )

                # Update layout to adjust the size and hide the legend
                crime_type_division_year_analysis = fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='#23272c',
                    paper_bgcolor='#23272c',
                    title_font=dict(size=24, color='#FFFFFF'),
                    font=dict(color='#FFFFFF'),
                    xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='#FFFFFF'),
                    yaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='#FFFFFF'),
                    showlegend=True,
                    xaxis_title="Year",
                    height=1500,  # Set the height
                    width=1500   # Set the width
                )




                #######################################  crimetype , divison , year analaysis ends here #####################


####################### crime type trends all over the years start session ###########################

                # Filter out 'Misc.' from CrimeType
                df_filtered = df[df['CrimeType'] != 'Misc.']

                # Aggregate data by Year and CrimeType
                agg_data = df_filtered.groupby(['Year', 'CrimeType'])['CrimeCount'].sum().reset_index()

                # Create a line chart
                fig = px.line(
                    agg_data,
                    x='Year',
                    y='CrimeCount',
                    color='CrimeType',
                    title='Crime Counts by Crime Type Over Years',
                    markers=True,  # Add markers to the line chart
                    line_shape='linear',  # Use linear line shape
                    hover_data={'CrimeCount': ':,'},  # Format hover data as a simple number
                    template = 'plotly_dark'
                )

                # Update layout to add axis labels
                crimetype_years_trend = fig.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Crime Count',
                )

                


####################### crime type trends all over the years end session ###########################



############################# divsions trends across years starts session ##############

                

                # Assuming df is your DataFrame and it's already loaded

                # Filter out 'Misc.' from CrimeType
                df_filtered = df[df['CrimeType'] != 'Misc.']

                # Aggregate data by Year, Division, and CrimeType
                agg_data = df_filtered.groupby(['Year', 'Division'])['CrimeCount'].sum().reset_index()

                # Create a line chart
                fig = px.line(
                    agg_data,
                    x='Year',
                    y='CrimeCount',
                    color='Division',
                    title='Crime Counts by Crime Type Over Years',
                    markers=True,  # Add markers to the line chart
                    line_shape='spline',  # Use spline line shape for smooth curves
                    hover_data={'CrimeCount': ':,'},  # Format hover data as a simple number
                )

                # Customize the appearance for a dark, cyberpunk-like theme
                divisions_years_trends = fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='#23272c',
                    paper_bgcolor='#23272c',
                    xaxis_title='Year',
                    yaxis_title='Crime Count',
                    title_font=dict(size=24, color='#FFFFFF'),
                    font=dict(color='#FFFFFF'),
                    xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='#FFFFFF'),
                    yaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='#FFFFFF'),
                )

            


###################### divisions trends across years ends here ###################



                ####################### crimes counts over years start session ##########################


                # Filter out 'Misc.' from CrimeType
                df_filtered = df[df['CrimeType'] != 'Misc.']

                # Aggregate data by Year
                agg_data = df_filtered.groupby(['Year'])['CrimeCount'].sum().reset_index()

                # Create a line chart with unique colors for each segment
                fig = go.Figure()

                # Define a list of colors
                colors = px.colors.qualitative.Set1

                # Add a trace for each year
                for i in range(1, len(agg_data)):
                    fig.add_trace(go.Scatter(
                        x=agg_data['Year'].iloc[i-1:i+1],
                        y=agg_data['CrimeCount'].iloc[i-1:i+1],
                        mode='lines+markers', 
                        line=dict(color=colors[i % len(colors)], width=4),
                        name=str(agg_data['Year'].iloc[i]),
                        text=[f'Total Crime Count: {count}' for count in agg_data['CrimeCount'].iloc[i-1:i+1]],
                        hoverinfo='text+x+y'
                    ))

                # Customize the appearance
                crimes_years_trends = fig.update_layout(
                    
                    title='Crime Counts Over Years',
                    template='plotly_dark',
                    plot_bgcolor='#23272c',
                    paper_bgcolor='#23272c',
                    xaxis_title='Year',
                    yaxis_title='Crime Count',
                    title_font=dict(size=24, color='#FFFFFF'),
                    font=dict(color='#FFFFFF'),
                    xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='#FFFFFF'),
                    yaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='#FFFFFF'),
                )

           


######################### crimes counts over years ends here ###########################

                # 2. Most Common Crimes
                crime_counts = df.groupby('CrimeType')['CrimeCount'].sum().sort_values(ascending=False).head(10)

                # Create a donut chart
                fig_most_common_crimes = px.pie(crime_counts.reset_index(), 
                                                names='CrimeType', 
                                                values='CrimeCount', 
                                                title='Most Common Crimes',
                                                hole=0.3)  # This creates a hole in the middle for a donut chart

                top_crime = crime_counts.idxmax()
                top_crime_count = crime_counts.max()
                analysis2 = f"The most common crime is '{top_crime}' with {top_crime_count} reported cases."

                # 3. Districts with Highest Crimes
                district_crime_counts = df.groupby('District')['CrimeCount'].sum().sort_values(ascending=False).head(10)
                fig_districts_with_highest_crimes = px.bar(district_crime_counts.reset_index(),
                                                           x='District', y='CrimeCount',
                                                           title='Districts with Highest Crimes',
                                                           color='District')
                highest_crime_district = district_crime_counts.idxmax()
                highest_crime_count = district_crime_counts.max()
                analysis3 = f"{highest_crime_district} district has the highest reported crimes with a total of {highest_crime_count} cases."

                # # 4. Crime Rate per Capita
                # df['CrimeRate'] = (df['CrimeCount'] / df['Population']) * 100000
                # average_crime_rate_per_district = df.groupby('District')['CrimeRate'].mean().sort_values(ascending=False).head(10)
                # fig_crime_rate_per_capita = px.bar(average_crime_rate_per_district.reset_index(),
                #                                    x='District', y='CrimeRate',
                #                                    title='Crime Rate per Capita (per 100,000 people)')
                # highest_crime_rate_district = average_crime_rate_per_district.idxmax()
                # highest_crime_rate = average_crime_rate_per_district.max()
                # analysis4 = f"{highest_crime_rate_district} district has the highest crime rate per capita with an average of {highest_crime_rate:.2f} crimes per 100,000 people."


                # 4. Crime Rate per Capita
                df['CrimeRate'] = (df['CrimeCount'] / df['Population']) * 100000
                average_crime_rate_per_district = df.groupby('District')['CrimeRate'].mean().sort_values(ascending=False).head(10)

                # Create a pie chart
                fig_crime_rate_per_capita = px.pie(average_crime_rate_per_district.reset_index(), 
                                                names='District', 
                                                values='CrimeRate',
                                                title='Crime Rate per Capita (per 100,000 people)',
                                                hole=0.3)  # Added a hole to make it a donut chart, you can remove the hole parameter for a full pie chart

                highest_crime_rate_district = average_crime_rate_per_district.idxmax()
                highest_crime_rate = average_crime_rate_per_district.max()
                analysis4 = f"{highest_crime_rate_district} district has the highest crime rate per capita with an average of {highest_crime_rate:.2f} crimes per 100,000 people."



###################### heatmap 1 start ######################
                # Identify the crime type with the highest number of reported incidents across all years
       
               
               # Pivot the data to create a matrix of CrimeType vs Year
                heatmap_data = df.pivot_table(index='CrimeType', columns='Year', values='CrimeCount', aggfunc='sum').fillna(0)

                # Create the heatmap
                fig_heatmap = px.imshow(heatmap_data,
                                        labels=dict(x="Year", y="Crime Type", color="Number of Crimes"),
                                        title="Number of Crimes by Crime Type and Year")


                
                # Create the heatmap
                fig_heatmap = px.imshow(heatmap_data,
                                        labels=dict(x="Year", y="Crime Type", color="Number of Crimes"),
                                        title="Number of Crimes by Crime Type and Year")

########################### heatmap 1 ends ###################################

                ################  HeatMap no 2 start
                # Identify the district with the highest number of crimes across all years
                highest_crime_district = heatmap_data.sum(axis=1).idxmax()
                total_crimes_highest_district = heatmap_data.sum(axis=1).max()

                # Identify the year with the highest number of crimes across all districts
                highest_crime_year = heatmap_data.sum().idxmax()
                total_crimes_highest_year = heatmap_data.sum().max()

                # Overall trend analysis
                first_year, last_year = heatmap_data.columns.min(), heatmap_data.columns.max()
                first_year_crimes, last_year_crimes = heatmap_data[first_year].sum(), heatmap_data[last_year].sum()
                if last_year_crimes > first_year_crimes:
                    trend = "increased"
                elif last_year_crimes < first_year_crimes:
                    trend = "decreased"
                else:
                    trend = "remained roughly stable"

                analysis_heatmap = (f"The district with the highest number of reported crimes across all years is '{highest_crime_district}'"
                                    f"with a total of {total_crimes_highest_district} crimes. "
                                    f"The year with the highest number of reported crimes across all districts is {highest_crime_year} "
                                    f"with a total of {total_crimes_highest_year} crimes. "
                                    f"From {first_year} to {last_year}, the overall number of reported crimes has {trend}.")
                analysis6 = analysis_heatmap
                # Pivot the dataframe to get years as columns, districts as index, and sum of crimes as values
                heatmap_data1 = df.pivot_table(index='District', columns='Year', values='CrimeCount', aggfunc='sum').fillna(0)

                # Create the heatmap using Plotly Express
                fig_heatmap1 = px.imshow(heatmap_data1, 
                                        labels=dict(x="Year", y="District", color="Number of Crimes"),
                                        title="Number of Crimes by District and Year")

                fig_heatmap1.update_xaxes(side="top")
                ######################## heat map 2 end #####################

############################# 3D plot start ############################
               # Aggregate data by year and crime type
                agg_data = df.groupby(['Year', 'CrimeType'])['CrimeCount'].sum().reset_index()

                # Convert CrimeType to a categorical type and then to integer for plotting
                agg_data['CrimeType_cat'] = agg_data['CrimeType'].astype('category').cat.codes

                # Create the 3D scatter plot
                fig = px.scatter_3d(agg_data, 
                                    x='Year', 
                                    y='CrimeType_cat', 
                                    z='CrimeCount',
                                    color='CrimeType', 
                                    size='CrimeCount',
                                    hover_name='CrimeType',
                                    title='3D Scatter Plot of Year, Crime Type, and Number of Cases')

                analysis_statement_for_3D_plot = ("""
This 3D scatter plot provides a comprehensive view of the crime landscape over the years. 
Each point on the graph represents a specific crime type in a particular year, with its height (z-axis) indicating the number of cases.
The color and size variations offer further insights into the type and magnitude of the crimes, respectively.
By observing the distribution and clustering of points, one can discern patterns, trends, and anomalies in the reported crimes over the years.
""")
                analysis7 = analysis_statement_for_3D_plot
               
############################### 3D plot end #############################

                # Convert the figures to JSON format
                

                fig_all_cities_on_map_json = all_cities_on_map.to_json()
                fig_most_badly_effected_district = most_badly_effected_district.to_json()
                fig_most_badly_effected_division = most_badly_effected_division.to_json()
                fig_crimes_category_ranking = crimes_category_ranking.to_json()
                fig_crime_type_division_year_analysis = crime_type_division_year_analysis.to_json()
                fig_crimetype_years_trend = crimetype_years_trend.to_json()
                fig_divisions_years_trends = divisions_years_trends.to_json()
                fig_crimes_years_trends = crimes_years_trends.to_json()


                fig001_json = fig001.to_json()
                fig002_json = fig002.to_json()
                fig003_json = fig003.to_json()
                fig004_json = fig004.to_json()
                fig005_json = fig005.to_json()
                fig006_json = fig006.to_json()
                fig007_json = fig007.to_json()
                fig008_json = fig008.to_json()
                fig009_json = fig009.to_json()

                fig_001_map_json = fig_001_map.to_json()
                fig_002_map_json = fig_002_map.to_json()
                fig_003_map_json = fig_003_map.to_json()
                fig_004_map_json = fig_004_map.to_json()
                fig_005_map_json = fig_005_map.to_json()
                fig_006_map_json = fig_006_map.to_json()
                fig_007_map_json = fig_007_map.to_json()
                fig_008_map_json = fig_008_map.to_json()
                fig_009_map_json = fig_009_map.to_json()
                
                fig1_json = fig_yearly_crime_trend.to_json()
                fig2_json = fig_most_common_crimes.to_json()
                fig3_json = fig_districts_with_highest_crimes.to_json()
                fig4_json = fig_crime_rate_per_capita.to_json()
                fig5_json = fig_heatmap.to_json()
                fig6_json = fig_heatmap1.to_json()
                fig7_json = fig.to_json()

            else:
                return HttpResponse("The uploaded CSV is missing some required columns.")

    return render(request, 'pak_crime_stats.html', {'fig_all_cities_on_map_json':fig_all_cities_on_map_json,'fig_most_badly_effected_district':fig_most_badly_effected_district,'fig_most_badly_effected_division':fig_most_badly_effected_division,'fig_crimes_category_ranking':fig_crimes_category_ranking,'fig_crime_type_division_year_analysis':fig_crime_type_division_year_analysis,'fig_crimetype_years_trend':fig_crimetype_years_trend,'fig_divisions_years_trends':fig_divisions_years_trends,'fig_crimes_years_trends':fig_crimes_years_trends,'figure_001_map': fig_001_map_json,'figure_002_map': fig_002_map_json,'figure_003_map': fig_003_map_json,'figure_004_map': fig_004_map_json,'figure_005_map': fig_005_map_json,'figure_006_map': fig_006_map_json,'figure_007_map': fig_007_map_json,'figure_008_map': fig_008_map_json,'figure_009_map': fig_009_map_json,'figure001': fig001_json,'figure002': fig002_json,'figure003': fig003_json,'figure004': fig004_json,'figure005': fig005_json,'figure006': fig006_json,'figure007': fig007_json,'figure008': fig008_json,'figure009': fig009_json,'figure1': fig1_json, 'figure2': fig2_json, 'figure3': fig3_json, 'figure4': fig4_json,'figure5':fig5_json,'figure6':fig6_json,'figure7':fig7_json,
                                                             'file_summary': file_summary, 'analysis1': analysis1, 'analysis2': analysis2, 'analysis3': analysis3, 'analysis4': analysis4,'analysis6': analysis6,'analysis7': analysis7})

   
   
#*************************************************





















def contactform(request):
    
    return render (request, "contact_form.html")

from django.shortcuts import render, redirect
from django.http import HttpResponse
from crimeregisterformdata.models import CrimeRegisterFormData

def register_crime(request):
    if request.method == "POST":
        # Extracting data from the request POST dictionary
        date = request.POST.get('inputDate')
        time = request.POST.get('inputTime')
        crime_type = request.POST.get('crimeType')
        location_city = request.POST.get('locationCity')
        latitude = request.POST.get('latitude')
        longitude = request.POST.get('longitude')
        crime_description = request.POST.get('crimeDescription')
        reported_type = request.POST.get('reportedType')
        status = request.POST.get('status')
        injuries = request.POST.get('injuries')
        victims = request.POST.get('victims')
        outcome = request.POST.get('outcome')
        news_resources = request.POST.get('newsResources')

        # Saving data into the model
        crime_data = CrimeRegisterFormData(
            date=date,
            time=time,
            crime_type=crime_type,
            location_city=location_city,
            latitude=latitude,
            longitude=longitude,
            crime_description=crime_description,
            reported_type=reported_type,
            status=status,
            injuries=injuries,
            victims=victims,
            outcome=outcome,
            news_resources=news_resources
        )
        crime_data.save()

        # Redirecting to the same page with a success message or to another page as needed
        return redirect('contactform')  # Assuming the name of the URL pattern for this view is 'register_crime'

    return render(request, 'contact_form.html')  # Replace 'path_to_your_template.html' with the path to your HTML template

################################ register crime data function end ##################################

#################### get in touch with us cocde starts heree ################

from contact_form.models import ContactFormEntry

def contact_form_submission(request):
    if request.method == 'POST':
        # Retrieve form data from POST request
        name = request.POST['name']
        email = request.POST['email']
        phone = request.POST['phone']
        message = request.POST['message']

        # Create a new instance of your model and save it to the database
        contact_entry = ContactFormEntry(
            name=name,
            email=email,
            phone=phone,
            message=message
        )
        contact_entry.save()

        # Optionally, you can perform additional actions here, such as sending emails.
        return redirect('myhomepage')  # Assuming the name of the URL pattern for this view is 'register_crime'

    return render(request, 'myhomepage.html') 

############# get  in touch with us ends here @##################




def angularjs(request):
    return render(request,'angular_js.html')


def secondtemplate(request):
    return render(request,'homedesign.html')




def operation_al_mizan(request):
    return render(request,'operation_al_mizan.html')


def operation_rah_e_haq(request):
    return render(request,'operation_rah_e_haq.html')


def operation_rah_e_nijaat(request):
    return render(request,'operation_rah_e_nijaat.html')


def operation_zarb_e_azb(request):
    return render(request,'operation_zarb_e_azb.html')


def operation_black_thunderstorm(request):
    return render(request,'operation_black_thunderstorm.html')


def operation_raddul_fasaad(request):
    return render(request,'operation_raddul_fasaad.html')


def karachi_operation(request):
    return render(request,'karachi_operation.html')


def operation_sherdil(request):
    return render(request,'operation_sherdil.html')