# from django.shortcuts import render
# from .models import CSVData
# import csv
# import spacy
# import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize


# nltk.download('stopwords')

# # def karachi(request):
# def generate_summary(content):
#     # Concatenate all sentences from the content into a single text
#     full_text = '\n'.join([' '.join(row) for row in content])

#     # Tokenize the full text into sentences
#     sentences = sent_tokenize(full_text)

#     # Preprocess the sentences by removing stop words and punctuation
#     stop_words = set(stopwords.words('english'))
#     preprocessed_sentences = []
#     for sentence in sentences:
#         word_tokens = word_tokenize(sentence.lower())
#         filtered_sentence = [word for word in word_tokens if word.isalnum() and word not in stop_words]
#         preprocessed_sentences.append(' '.join(filtered_sentence))

#     # Calculate the TF-IDF scores for the sentences
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
#     scores = tfidf_matrix.sum(axis=1)

#     # Find the index of the sentence with the highest TF-IDF score
#     idx = scores.argmax()

#     # Get the original sentence with the highest TF-IDF score
#     summary = sentences[idx]

#     return summary

# def predict_crime_occurrence(content):
#     # Concatenate all sentences from the content into a single text
#     full_text = '\n'.join([' '.join(row) for row in content])

#     # Perform Sentiment Analysis on the full text
#     blob = TextBlob(full_text)
#     sentiment_score = blob.sentiment.polarity

#     # Determine whether the content indicates potential crime occurrence or not
#     # For this example, we'll assume that a negative sentiment indicates potential crime.
#     # You can adjust this threshold based on your specific use case and data.
#     crime_prediction = "Crime may occur." if sentiment_score < 0 else "No indication of crime."

#     return crime_prediction

# def karachi(request):
#     data = CSVData.objects.all()

#     for item in data:
#         if item.csv_name == 'karachi.csv':  # Replace 'csv2.csv' with the name of your second CSV file
#             content = []
#             extracted_entities = []  # Initialize inside the loop to reset for each row
#             with open(item.file.path, 'r', encoding='utf-8') as file:
#                 csv_reader = csv.reader(file, delimiter=',')  # Adjust the delimiter as per your CSV
#                 for row in csv_reader:
#                     content.append(row)
#             item.content = content

#             Locations = []
#             Dates = []
#             Times = []
#             Persons = []
#             Others = []
#             Summaries = []  # List to store the summaries for each CSV file
#             CrimePredictions = []  # List to store the crime predictions for each CSV file

#             # Highlight date, time, and location entities in each row of the CSV content
#             for row in content:
#                 for sentence in row:
#                     # Tokenize the sentence
#                     tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#                     sentences = tokenizer.tokenize(sentence)

#                     # Process each sentence with spaCy
#                     nlp = spacy.load('en_core_web_sm')
#                     summary = generate_summary(content)
#                     Summaries.append(summary)  # Append the summary to the list

#                     # Perform Sentiment Analysis and get crime prediction
#                     crime_prediction = predict_crime_occurrence(content)
#                     CrimePredictions.append(crime_prediction)
                   
#                     for sent in sentences:
#                         doc = nlp(sent)
#                         for ent in doc.ents:
#                             extracted_entities.append({"text": ent.text, "label": ent.label_})

#                             # Add entities to their respective variables
#                             if ent.label_ == 'GPE':
#                                 Locations.append(ent.text)
#                             elif ent.label_ == 'DATE':
#                                 Dates.append(ent.text)
#                             elif ent.label_ == 'TIME':
#                                 Times.append(ent.text)
#                             elif ent.label_ == 'PERSON':
#                                 Persons.append(ent.text)
#                             else:
#                                 Others.append({"text": ent.text, "label": ent.label_})

#             item.entities = extracted_entities

#             # Render the template after processing all items
#             return render(request, 'templates/karachi.html', {
#                 'item': item,
#                 'Locations': Locations,
#                 'Dates': Dates,
#                 'Times': Times,
#                 'Persons': Persons,
#                 'Others': Others,
#                 'Summaries': Summaries,
#                 'CrimePredictions': CrimePredictions,
#             })

#     # Return an empty response if the CSV file is not found
#     return render(request, 'templates/karachi.html', {
#         'data': data,
#     })
