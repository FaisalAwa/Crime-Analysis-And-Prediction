from django.shortcuts import render,HttpResponse,redirect,redirect,get_object_or_404
from django.contrib.auth.models import  User
from django.contrib.auth import authenticate,login,logout
from django.shortcuts import render
from requests import request
from .models import CSVData, University,Image
import csv
import nltk
import spacy
import codecs
import nltk.data
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from textblob import TextBlob
from django.contrib import messages
import csv
import spacy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from io import BytesIO
import base64
from django.http import HttpResponse,JsonResponse
from .models import CSVData
from spacy.lang.en import English
from django.shortcuts import render
from textblob.sentiments import NaiveBayesAnalyzer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from subject_object_extraction import findSVOs
import seaborn as sns
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.en import English
from .models import UserComment, register_table
from datetime import datetime
from django.contrib.auth.decorators import login_required
from django.core.mail import EmailMessage
import random
# Create your views here.

def home(request):
    return render(request, 'home.html')

def splash(request):
     return render(request, 'splash.html')
 
def info(request):
     return render(request, 'info.html')

def landing(request):
     return render(request, 'landing.html') 
 
def abc(request):
     return render(request, 'abc.html') 
 
def help(request):
     return render(request, 'help.html') 
 
def main(request):
    return render(request, 'main.html')

def about(request):
    return render(request, 'about.html')

def team(request):
    return render(request, 'team.html')

def services(request):
    return render(request, 'services.html')

def contact(request):
    return render(request, 'contact.html')




def signup(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')

        # Check if the username already exists
        if User.objects.filter(username=uname).exists():
            error_message = "Username already exists. Please choose another username."
            return render(request, 'signup.html', {'error_message': error_message})

        if pass1 != pass2:
            return HttpResponse("Your password and confirm password are not the same!!")
        else:
            my_user = User.objects.create_user(username=uname, email=email, password=pass1, first_name=first_name, last_name=last_name)
            my_user.save()
            return redirect('signin')

    return render(request, 'signup.html')

        
        
 


from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from datetime import datetime

def signin(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            if user.is_superuser:
                return redirect('admin') 
            else:
                res = redirect('main')
                if "rememberme" in request.POST:
                    res.set_cookie("user_id", user.id)
                    res.set_cookie("date_login", datetime.now())
                return res
        else:
            # Redirect to the login error page with a message
            return render(request, "login_error.html")

    return render(request, 'signin.html')


@login_required
def user_dashboard(request):
    context = {}
    check = register_table.objects.filter(user__id=request.user.id)
    
    if len(check) > 0:
        data = register_table.objects.get(user__id=request.user.id)
        context["data"] = data
    
    # Redirect to the user_dashboard page
    return render(request,"user_dashboard.html",context)


@login_required
def user_logout(request):
    logout(request)
    res =  redirect('home')
    res.delete_cookie("user_id")
    res.delete_cookie("date_login")
    return res


def edit_profile(request):
    context = {}
    data = None  # Define data with None initially
    check = register_table.objects.filter(user__id=request.user.id)
    if len(check) > 0:
        data = register_table.objects.get(user__id=request.user.id)
        context["data"] = data   
    if request.method == "POST":
        fn = request.POST["fname"]
        ln = request.POST["lname"]
        em = request.POST["email"]

        usr = User.objects.get(id=request.user.id)
        usr.first_name = fn
        usr.last_name = ln
        usr.email = em
        usr.save()

        if data is not None:  # Check if data is not None before accessing its attributes
            if "image" in request.FILES:
                img = request.FILES["image"]
                data.profile_pic = img
                data.save()

        context["status"] = "Changes Saved Successfully"
    return render(request, "edit_profile.html", context)



def sendemail(request):
    context = {}
    ch = register_table.objects.filter(user__id=request.user.id)
    if len(ch)>0:
        data = register_table.objects.get(user__id=request.user.id)
        context["data"] = data

    if request.method=="POST":
    
        rec = request.POST["to"].split(",")
        print(rec)
        sub = request.POST["sub"]
        msz = request.POST["msz"]

        try:
            em = EmailMessage(sub,msz,to=rec)
            em.send()
            context["status"] = "Email Sent"
            context["cls"] = "alert-success"
        except:
            context["status"] = "Could not Send, Please check Internet Connection / Email Address"
            context["cls"] = "alert-danger"
    return render(request,"sendemail.html",context  )


def forgot_password(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        
        try:
            user = User.objects.get(username=username, email=email)
            password = user.password  # This will retrieve the hashed password
            
            # You should implement your email sending logic here
            # Use the 'send_mail' function to send the password to the user's email
            # Example:
            # send_mail('Your Password', f'Your password is: {password}', 'from@example.com', [email])
            
            return render(request, 'forgot_password_success.html')
        except User.DoesNotExist:
            return render(request, 'forgot_password.html', {'error': 'User not found.'})
    else:
        return render(request, 'forgot_password.html')



def password_reset(request):
     return render(request, 'password_reset.html')
 
def open(request):
     return render(request, 'open.html')
# Create your views here.





def csv_data(request):
    data = CSVData.objects.all()

    # Load spaCy model outside of the loops
    nlp = spacy.load('en_core_web_sm')

    for item in data:
        content = []
        extracted_entities = []  # List to store the extracted entities
        
        with codecs.open(item.file.path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file, delimiter='\t')  # Specify the delimiter used in your CSV file
            for row in csv_reader:
                content.append(row)
        item.content = content

        # Highlight date, time, and location entities in each row of the CSV content
        for row in content:
            for sentence in row:
                tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                sentences = tokenizer.tokenize(sentence)

                for sent in sentences:
                    doc = nlp(sent)
                    for ent in doc.ents:
                        extracted_entities.append({"text": ent.text, "label": ent.label_})

        item.entities = extracted_entities

    return render(request, 'csvapp/csv_data.html', {'data': data})


def comsats(request):
    data = CSVData.objects.filter(csv_name='comsats.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['Comsats'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['Comsats']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2)
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'comsats.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')



    


def uniwah(request):
    data = CSVData.objects.filter(csv_name='uniwah.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['Uniwah'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['Uniwah']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2)
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'uniwah.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')


def nust(request):
    data = CSVData.objects.filter(csv_name='nust.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['NUST'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['NUST']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2)
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'nust.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')
def air(request):
    data = CSVData.objects.filter(csv_name='air.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['AIR'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['AIR']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

        # Create a DataFrame to plot sentiment counts
        df2 = pd.DataFrame(dict(x=pos_neg))
        ax = sns.countplot(x="x", data=df2)
        plt.title('Sentiment Analysis')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')

        # Save the plot to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()

        # Store sentiment and graph in a dictionary for each item
        sentiment_data.append({
            'content': X,
            'sentiment': sentiment,
            'sentiment_graph': sentiment_graph,
        })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'air.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count, 'sentiment_graph': sentiment_graph})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')



def gift(request):
    data = CSVData.objects.filter(csv_name='gift.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['GIFT'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['GIFT']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2)
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'gift.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')

def iiui(request):
    data = CSVData.objects.filter(csv_name='iiui.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['IIUI'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['IIUI']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2)
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'iiui.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')

def lums(request):
    data = CSVData.objects.filter(csv_name='lums.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['LUMS'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['LUMS']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2)
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'lums.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')


def pieas(request):
    data = CSVData.objects.filter(csv_name='pieas.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['PIEAS'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['PIEAS']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2)
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'pieas.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')

def uet(request):
    data = CSVData.objects.filter(csv_name='uet.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['Uet'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['Uet']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2)
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'uet.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')
def cust(request):
    data = CSVData.objects.filter(csv_name='cust.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['CUST'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['CUST']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2)
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'cust.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')

def hitec(request):
    data = CSVData.objects.filter(csv_name='hitec.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['HITEC'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['HITEC']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2)
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'hitec.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')

def riphah(request):
    data = CSVData.objects.filter(csv_name='riphah.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['RIPHAH'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['RIPHAH']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2)
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'riphah.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')


def bahria(request):
    data = CSVData.objects.filter(csv_name='bahria.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['BAHRIA'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['BAHRIA']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2)
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'bahria.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')



def kust(request):
    data = CSVData.objects.filter(csv_name='kust.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['KUST'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['KUST']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2)
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'kust.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')


def quaid(request):
    data = CSVData.objects.filter(csv_name='quaid.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        for i in range(len(data)):
            X = data['Quaid e Azam'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['Quaid e Azam']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2)
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'quaid.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')



def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment


def ai_work(request):
    # Add your view logic here
    return render(request, 'ai_work.html') 

def universities_dashboard(request):
    universities = University.objects.all()
    return render(request, 'universities_dashboard.html', {'universities': universities})

def uni_home(request):
    universities = University.objects.all()
    return render(request, 'uni_home.html', {'universities': universities})

from django.http import JsonResponse
from django.contrib.auth.models import User  # Import the User model

def get_user_registration_count(request):
    user_count = User.objects.count()
    data = {'count': user_count}
    return JsonResponse(data)

from django.http import JsonResponse
from .models import University

def get_total_university_count(request):
    university_count = University.objects.count()
    data = {
        'count': university_count
    }
    return JsonResponse(data)

from django.http import JsonResponse
from APP1.models import CSVData  # Import your CSVData model

def get_total_csv_data_count(request):
    total_count = CSVData.objects.count()
    data={
        'count':total_count
        }
    return JsonResponse(data)


from django.http import JsonResponse
from .models import University

def get_total_university_count(request):
    university_count = University.objects.count()
    data = {
        'count': university_count
    }
    return JsonResponse(data)

from django.http import JsonResponse
from .models import UserComment

def get_total_comment_count(request):
    comment_count = UserComment.objects.count()
    data = {
        'count': comment_count
    }
    return JsonResponse(data)

from django.http import JsonResponse
from .models import Program  # Import the Program model or adjust it to your actual model

def get_total_program_count(request):
    program_count = Program.objects.count()  # Adjust the model as needed
    data = {
        'count': program_count
    }
    return JsonResponse(data)

from django.http import JsonResponse
from .models import MCQResponse

def get_total_mcq_count(request):
    mcq_count = MCQResponse.objects.count()
    data = {
        'count': mcq_count
    }
    return JsonResponse(data)





def saveForm(request):
     if request.method == 'POST':
        name = request.POST['name']
        email = request.POST['email']
        comment= request.POST['comment']
        
        # Create and save a new Comments instance
        ad = UserComment(name=name, email=email, comment=comment)
        ad.save()
     return render(request, 'main.html')

from django.shortcuts import render, redirect
from .models import MCQResponse
def question(request):
     if request.method == 'POST':
        user_name = request.user.username
        question_1 = request.POST['question_1']
        question_2 = request.POST['question_2']
        question_3 = request.POST['question_3']
        question_4 = request.POST['question_4']
        question_5 = request.POST['question_5']
        question_6 = request.POST['question_6']
        question_7 = request.POST['question_7']
        question_8 = request.POST['question_8']
        question_9 = request.POST['question_9']
        question_10 = request.POST['question_10']
        question_11 = request.POST['question_11']
        question_12 = request.POST['question_12']
        question_13 = request.POST['question_13']
        question_14 = request.POST['question_14']
        question_15 = request.POST['question_15']
        question_16 = request.POST['question_16']
        question_17 = request.POST['question_17']
        question_18 = request.POST['question_18']
        question_19 = request.POST['question_19']
        
        mcq_response = MCQResponse(
            user_name=user_name,
            question_1=question_1,
            question_2=question_2,
            question_3=question_3,
            question_4=question_4,
            question_5=question_5,
            question_6=question_6,
            question_7=question_7,
            question_8=question_8,
            question_9=question_9,
            question_10=question_10,
            question_11=question_11,
            question_12=question_12,
            question_13=question_13,
            question_14=question_14,
            question_15=question_15,
            question_16=question_16,
            question_17=question_17,
            question_18=question_18,
            question_19=question_19
            
        )
        mcq_response.save()
        programs = Program.objects.all()
        recommended_program = recommend_program(request.POST)
        return render(request, 'interest.html', {'recommended_program': recommended_program,'programs': programs})
     return render(request,'question.html')

 
def recommend_program(user_responses):
    program_criteria = {
        'Computer Science': {
            'question_1': ['Yes, working with a diverse team and creating a real-world impact is a career goal of mine'],
            'question_2': ['I am more interested in working with software and digital systems.'],
            'question_3': ['No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['No, it does not excite me.'],
            'question_5': ['No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.', 'I have not explored this much yet.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.','Somewhat excited, but I have other interests too'],
            'question_11': ['Business does not appeal to me.','It is a consideration, but not my primary interest.'],
            'question_12': ['I would not be committed and would not focus on innovative approaches or market trends.','I am unsure about my level of commitment and approach to innovation.'],
            'question_13': ['I am not sure.','I am exploring different options.'],
            'question_14': ['No, I have other academic or career interests that I am more inclined towards','I am not sure if this aligns with my career goals'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['Aviation does not captivate me.','I have not delved into aviation much yet.','It is intriguing, but not my sole focus.'],
            'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],

        },   
        'Cyber Security': {
             'question_1': ['I am still exploring my career goals and need more information about the benefits of teamwork in achieving real-world impact'],
             'question_2': ['I am not sure.'],
            'question_3': ['Yes, I am passionate about ensuring the safety and security of digital assets and information'],
            'question_4': ['Yes, I am passionate about cybersecurity.'],
            'question_5': ['No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.', 'I have not explored this much yet.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.','Somewhat excited, but I have other interests too'],
            'question_11': ['Business does not appeal to me.','It is a consideration, but not my primary interest.'],
            'question_12': ['I would not be committed and would not focus on innovative approaches or market trends.','I am unsure about my level of commitment and approach to innovation.'],
            'question_13': ['I am not sure.','I am exploring different options.'],
            'question_14': ['No, I have other academic or career interests that I am more inclined towards','I am not sure if this aligns with my career goals'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['Aviation does not captivate me.','I have not delved into aviation much yet.','It is intriguing, but not my sole focus.'],
            'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
        },
        'AVM': {
            'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I prefer working with my hands and physical systems.'],
            'question_3': ['No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['No, it does not excite me.'],
            'question_5': ['No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.', 'I have not explored this much yet.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.','Somewhat excited, but I have other interests too'],
            'question_11': ['Business does not appeal to me.','It is a consideration, but not my primary interest.'],
            'question_12': ['I would not be committed and would not focus on innovative approaches or market trends.','I am unsure about my level of commitment and approach to innovation.'],
            'question_13': ['I am not sure.','I am exploring different options.'],
            'question_14': ['No, I have other academic or career interests that I am more inclined towards','I am not sure if this aligns with my career goals'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
             'question_16': ['Innovations in airline fleet management and aviation technology',
                                'Managing and optimizing airport facilities for efficient operations and customer satisfaction',
                                'Developing effective marketing strategies for the aviation industry'],
             'question_17': ['Aviation is a dream of mine.'],
             'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],

            
            
             },
        'BBA': {
            'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
            'question_3': ['No, I believe my independent efforts can create a more substantial and direct impact on projects','No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['No, it does not excite me.'],
            'question_5': ['No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
             'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.', 'I have not explored this much yet.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.','Somewhat excited, but I have other interests too'],
            'question_11': ['Absolutely, I am passionate about business'],
            'question_12': ['I would be fully committed and eager to explore innovative approaches and market trends'],
            'question_13': ['I am not sure.','I am exploring different options.'],
            'question_14': ['No, I have other academic or career interests that I am more inclined towards','I am not sure if this aligns with my career goals'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
             'question_16': ['I am not interested in aviation'],
              'question_17': ['Aviation does not captivate me.','I have not delved into aviation much yet.','It is intriguing, but not my sole focus.'],
             'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],

        },
        'Electrical': {
            'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
            'question_3': ['No, I believe my independent efforts can create a more substantial and direct impact on projects','No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['No, it does not excite me.'],
             'question_5': ['No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
             'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.', 'I have not explored this much yet.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.','Somewhat excited, but I have other interests too'],
            'question_11': ['Business does not appeal to me.','It is a consideration, but not my primary interest.'],
            'question_12': ['I would not be committed and would not focus on innovative approaches or market trends.','I am unsure about my level of commitment and approach to innovation.'],
            'question_13': ['I am fascinated by technology and innovation.'],
            'question_14': ['Yes, I am fascinated by electrical engineering and the vast array of applications it offers in technology and industry'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
             'question_17': ['Aviation does not captivate me.','I have not delved into aviation much yet.','It is intriguing, but not my sole focus.'],
            'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],


              },
        'Mechanical': {
            'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
            'question_3': ['No, I believe my independent efforts can create a more substantial and direct impact on projects','No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['No, it does not excite me.'],
            'question_5': ['No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.', 'I have not explored this much yet.'],
            'question_9': ['Yes, I am fascinated by the principles of mechanics and engineering, and I enjoy designing and building machines'],
            'question_10': ['Extremely excited; it is my passion.'],
            'question_11': ['Business does not appeal to me.','It is a consideration, but not my primary interest.'],
            'question_12': ['I would not be committed and would not focus on innovative approaches or market trends.','I am unsure about my level of commitment and approach to innovation.'],
            'question_13': ['I am exploring different options.','I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals','No, I have other academic or career interests that I am more inclined towards'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
             'question_16': ['I am not interested in aviation'],
             'question_17': ['Aviation does not captivate me.','I have not delved into aviation much yet.','It is intriguing, but not my sole focus.'],
            'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
 },
        'English': {
            'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
            'question_3': ['No, I believe my independent efforts can create a more substantial and direct impact on projects','No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['No, it does not excite me.'],
            'question_5': ['Yes, I love playing with words and expressing my thoughts creatively'],
            'question_6': ['Literature and storytelling are my passions.'],
            'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.', 'I have not explored this much yet.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
             'question_10': ['Not excited at all.','Somewhat excited, but I have other interests too'],
            'question_11': ['Business does not appeal to me.','It is a consideration, but not my primary interest.'],
            'question_12': ['I would not be committed and would not focus on innovative approaches or market trends.','I am unsure about my level of commitment and approach to innovation.'],
            'question_13': ['I am not sure.','I am exploring different options.'],
            'question_14': ['No, I have other academic or career interests that I am more inclined towards','I am not sure if this aligns with my career goals'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
             'question_17': ['Aviation does not captivate me.','I have not delved into aviation much yet.','It is intriguing, but not my sole focus.'],
            'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
       
            
        },
        'Chemistry': {
         'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
            'question_3': ['No, I believe my independent efforts can create a more substantial and direct impact on projects','No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['No, it does not excite me.'],
            'question_5': ['No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
           'question_7': ['Yes, I am passionate about chemistry and its role in shaping our understanding of the world'],
            'question_8': ['Absolutely, I find it incredibly intriguing.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
             'question_10': ['Not excited at all.','Somewhat excited, but I have other interests too'],
            'question_11': ['Business does not appeal to me.','It is a consideration, but not my primary interest.'],
            'question_12': ['I would not be committed and would not focus on innovative approaches or market trends.','I am unsure about my level of commitment and approach to innovation.'],
            'question_13': ['I am not sure.','I am exploring different options.'],
            'question_14': ['No, I have other academic or career interests that I am more inclined towards','I am not sure if this aligns with my career goals'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
             'question_17': ['Aviation does not captivate me.','I have not delved into aviation much yet.','It is intriguing, but not my sole focus.'],
            'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'], },
        
        
        'AVM and BBA': {
                 'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
          
                'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
                 'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
                 'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
                 'question_6': ['Literature does not interest me.'],
                 'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
                 'question_8': ['No, it does not captivate my interest.'],
                 'question_9': ['I am not sure if this aligns with my career goals; could you provide more information on the subjects and potential career paths?',
                                'No, I have other academic or career interests that I am more inclined towards'],
                 'question_10': ['Not excited at all.'],
                'question_11': ['Absolutely, I AMm passionate about business.'],
                
                'question_12':['I would be fully committed and eager to explore innovative approaches and market trends.'],
                'question_13': ['Iam exploring different options.',
                                'I am not sure.'],
                'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
                'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
                
                'question_16': ['Innovations in airline fleet management and aviation technology',
                                'Managing and optimizing airport facilities for efficient operations and customer satisfaction',
                                'Developing effective marketing strategies for the aviation industry'],
                'question_17': ['Aviation is a dream of mine.'],
                  'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
                
        },
          
          
           
        'AVM and Computer Science': {
                 'question_1': ['Yes, working with a diverse team and creating a real-world impact is a career goal of mine'],
                 'question_2': ['I am more interested in working with software and digital systems.'],
           
                'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
                 'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
                 'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
                 'question_6': ['Literature does not interest me.'],
                 'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
                 'question_8': ['No, it does not captivate my interest.'],
                 'question_9': ['I am not sure if this aligns with my career goals; could you provide more information on the subjects and potential career paths?',
                                'No, I have other academic or career interests that I am more inclined towards'],
                 'question_10': ['Not excited at all.'],
                'question_11': ['Business does not appeal to me.'],
                
                'question_12':['I would not be committed and would nit focus on innovative approaches or market trends.'],
                'question_13': ['Iam exploring different options.',
                                'I am not sure.'],
                'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
                'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
                
                'question_16': ['Innovations in airline fleet management and aviation technology',
                                'Managing and optimizing airport facilities for efficient operations and customer satisfaction',
                                'Developing effective marketing strategies for the aviation industry'],
                'question_17': ['Aviation is a dream of mine.'],
                  'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
                
        },
            
             'AVM and Cyber Security': {
                'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
          
                'question_3': ['Yes, I am passionate about ensuring the safety and security of digital assets and information'],
                'question_4': ['Yes, I am passionate about cybersecurity.'],
                 'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
                 'question_6': ['Literature does not interest me.'],
                 'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
                 'question_8': ['No, it does not captivate my interest.'],
                 'question_9': ['I am not sure if this aligns with my career goals; could you provide more information on the subjects and potential career paths?',
                                'No, I have other academic or career interests that I am more inclined towards'],
                 'question_10': ['Not excited at all.'],
                'question_11': ['Business does not appeal to me.'],
                
                'question_12':['I would not be committed and would nit focus on innovative approaches or market trends.'],
                'question_13': ['Iam exploring different options.',
                                'I am not sure.'],
                'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
                'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
                
                'question_16': ['Innovations in airline fleet management and aviation technology',
                                'Managing and optimizing airport facilities for efficient operations and customer satisfaction',
                                'Developing effective marketing strategies for the aviation industry'],
                'question_17': ['Aviation is a dream of mine.'],
                  'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
                
        },

            'AVM and English': {
                'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
          
                'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
                 'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
                  'question_5': ['Yes, I love playing with words and expressing my thoughts creatively'],
            'question_6': ['Literature and storytelling are my passions.'],
                 'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
                 'question_8': ['No, it does not captivate my interest.'],
                 'question_9': ['I am not sure if this aligns with my career goals; could you provide more information on the subjects and potential career paths?',
                                'No, I have other academic or career interests that I am more inclined towards'],
                 'question_10': ['Not excited at all.'],
                'question_11': ['Business does not appeal to me.'],
                
                'question_12':['I would not be committed and would nit focus on innovative approaches or market trends.'],
                'question_13': ['Iam exploring different options.',
                                'I am not sure.'],
                'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
                'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
                
                'question_16': ['Innovations in airline fleet management and aviation technology',
                                'Managing and optimizing airport facilities for efficient operations and customer satisfaction',
                                'Developing effective marketing strategies for the aviation industry'],
                'question_17': ['Aviation is a dream of mine.'],
                  'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
                
        },
          
          'AVM and Chemistry': {
                'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
          
                'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
                 'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
                  'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
                 'question_6': ['Literature does not interest me.'],
                  'question_7': ['Yes, I am passionate about chemistry and its role in shaping our understanding of the world'],
                 'question_8': ['Absolutely, I find it incredibly intriguing.'],
                 'question_9': ['I am not sure if this aligns with my career goals; could you provide more information on the subjects and potential career paths?',
                                'No, I have other academic or career interests that I am more inclined towards'],
                 'question_10': ['Not excited at all.'],
                'question_11': ['Business does not appeal to me.'],
                
                'question_12':['I would not be committed and would nit focus on innovative approaches or market trends.'],
                'question_13': ['Iam exploring different options.',
                                'I am not sure.'],
                'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
                'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
                
                'question_16': ['Innovations in airline fleet management and aviation technology',
                                'Managing and optimizing airport facilities for efficient operations and customer satisfaction',
                                'Developing effective marketing strategies for the aviation industry'],
                'question_17': ['Aviation is a dream of mine.'],
                  'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
                
        },
          
          'AVM and Electrical': {
                'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
          
                'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
                 'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
                  'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
                 'question_6': ['Literature does not interest me.'],
               'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
                 'question_8': ['No, it does not captivate my interest.'],
                 'question_9': ['I am not sure if this aligns with my career goals; could you provide more information on the subjects and potential career paths?',
                                'No, I have other academic or career interests that I am more inclined towards'],
                 'question_10': ['Not excited at all.'],
                'question_11': ['Business does not appeal to me.'],
                
                'question_12':['I would not be committed and would nit focus on innovative approaches or market trends.'],
                'question_13': ['I am fascinated by technology and innovation.'],
                'question_14': ['Yes, I am fascinated by electrical engineering and the vast array of applications it offers in technology and industry'],
           
                'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
                
                'question_16': ['Innovations in airline fleet management and aviation technology',
                                'Managing and optimizing airport facilities for efficient operations and customer satisfaction',
                                'Developing effective marketing strategies for the aviation industry'],
                'question_17': ['Aviation is a dream of mine.'],
                  'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
                
        },
          
             'AVM and Mechanical': {
                'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
          
                'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
                 'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
                  'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
                 'question_6': ['Literature does not interest me.'],
               'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
                 'question_8': ['No, it does not captivate my interest.'],
                 'question_9': ['Yes, I am fascinated by the principles of mechanics and engineering, and I enjoy designing and building machines'],
            'question_10': ['Extremely excited; it is my passion.'],
                'question_11': ['Business does not appeal to me.'],
                
                'question_12':['I would not be committed and would nit focus on innovative approaches or market trends.'],
               'question_13': ['Iam exploring different options.',
                                'I am not sure.'],
                'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
                'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
                
                'question_16': ['Innovations in airline fleet management and aviation technology',
                                'Managing and optimizing airport facilities for efficient operations and customer satisfaction',
                                'Developing effective marketing strategies for the aviation industry'],
                'question_17': ['Aviation is a dream of mine.'],
                  'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
                
        },
                
        'Computer Science and Cyber Security': {
            'question_1': ['Yes, working with a diverse team and creating a real-world impact is a career goal of mine'],
            'question_2': ['I am more interested in working with software and digital systems.'],
            'question_3': ['Yes, I am passionate about ensuring the safety and security of digital assets and information'],
            'question_4': ['Yes, I am passionate about cybersecurity.'],
            'question_5': ['No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
            'question_11': ['Business does not appeal to me.'],
            'question_12': ['I would not be committed and would not focus on innovative approaches or market trends.'],
            'question_13': ['I am not sure.'],
            'question_14': ['No, I have other academic or career interests that Iam more inclined towards'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },
        
          'Computer Science and English': {
            'question_1': ['Yes, working with a diverse team and creating a real-world impact is a career goal of mine'],
            'question_2': ['I am more interested in working with software and digital systems.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
            'question_5': ['Yes, I love playing with words and expressing my thoughts creatively'],
            'question_6': ['Literature and storytelling are my passions.'],
            'question_7': ['','No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
            'question_11': ['Business does not appeal to me.'],
            'question_12': ['I am unsure about my level of commitment and approach to innovation.',
                            'I would not be committed and would not focus on innovative approaches or market trends.'],
            'question_13': ['I am exploring different options..',
                            'I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },

           'Computer Science and Chemistry': {
            'question_1': ['Yes, working with a diverse team and creating a real-world impact is a career goal of mine'],
            'question_2': ['I am more interested in working with software and digital systems.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
           'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['Yes, I am passionate about chemistry and its role in shaping our understanding of the world'],
            'question_8': ['Absolutely, I find it incredibly intriguing.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
            'question_11': ['Business does not appeal to me.'],
            'question_12': ['I am unsure about my level of commitment and approach to innovation.',
                            'I would not be committed and would not focus on innovative approaches or market trends.'],
            'question_13': ['I am exploring different options..',
                            'I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },
           
             'Computer Science and BBA': {
            'question_1': ['Yes, working with a diverse team and creating a real-world impact is a career goal of mine'],
            'question_2': ['I am more interested in working with software and digital systems.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
           'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['','No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
            'question_11': ['Absolutely, I AMm passionate about business.'],
                
           'question_12':['I would be fully committed and eager to explore innovative approaches and market trends.'],
          'question_13': ['I am exploring different options..',
                            'I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },
             
               'Computer Science and Electrical': {
            'question_1': ['Yes, working with a diverse team and creating a real-world impact is a career goal of mine'],
            'question_2': ['I am more interested in working with software and digital systems.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
           'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['','No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
             'question_11': ['Business does not appeal to me.'],
            'question_12': ['I am unsure about my level of commitment and approach to innovation.',
                            'I would not be committed and would not focus on innovative approaches or market trends.'],
          
          'question_13': ['I am fascinated by technology and innovation.'],
            'question_14': ['Yes, I am fascinated by electrical engineering and the vast array of applications it offers in technology and industry'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },
         
          'Computer Science and Mechanical': {
            'question_1': ['Yes, working with a diverse team and creating a real-world impact is a career goal of mine'],
            'question_2': ['I am more interested in working with software and digital systems.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
           'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['','No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
            'question_9': ['Yes, I am fascinated by the principles of mechanics and engineering, and I enjoy designing and building machines'],
            'question_10': ['Extremely excited; it is my passion.'],
             'question_11': ['Business does not appeal to me.'],
            'question_12': ['I am unsure about my level of commitment and approach to innovation.',
                            'I would not be committed and would not focus on innovative approaches or market trends.'],
          
        'question_13': ['I am exploring different options..',
                            'I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],  'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },   
          
          'Cyber Security and BBA': {
            'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
            'question_3': ['Yes, I am passionate about ensuring the safety and security of digital assets and information'],
            'question_4': ['Yes, I am passionate about cybersecurity.'],
            'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['','No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
            'question_11': ['Absolutely, I am passionate about business.'],
            'question_12': ['I would be fully committed and eager to explore innovative approaches and market trends.'],
            'question_13': ['I am exploring different options.',
                            'I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },   
          
            'Cyber Security and English': {
            'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
            'question_3': ['Yes, I am passionate about ensuring the safety and security of digital assets and information'],
            'question_4': ['Yes, I am passionate about cybersecurity.'],
            'question_5': ['Yes, I love playing with words and expressing my thoughts creatively'],
            'question_6': ['Literature and storytelling are my passions.'],
            'question_7': ['','No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
            'question_11': ['Business does not appeal to me.'],
            'question_12': ['I am unsure about my level of commitment and approach to innovation.',
                            'I would not be committed and would not focus on innovative approaches or market trends.'],
            'question_13': ['I am exploring different options.',
                            'I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },

            'Cyber Security and Chemistry': {
            'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
            'question_3': ['Yes, I am passionate about ensuring the safety and security of digital assets and information'],
            'question_4': ['Yes, I am passionate about cybersecurity.'],
              'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
         
            'question_7': ['','Yes, I am passionate about chemistry and its role in shaping our understanding of the world'],
            'question_8': ['Absolutely, I find it incredibly intriguing.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
            'question_11': ['Business does not appeal to me.'],
            'question_12': ['I am unsure about my level of commitment and approach to innovation.',
                            'I would not be committed and would not focus on innovative approaches or market trends.'],
            'question_13': ['I am exploring different options.',
                            'I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },
            
              'Cyber Security and Electrical': {
            'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
            'question_3': ['Yes, I am passionate about ensuring the safety and security of digital assets and information'],
            'question_4': ['Yes, I am passionate about cybersecurity.'],
              'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
         
            'question_7': ['','No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
            'question_11': ['Business does not appeal to me.'],
            'question_12': ['I am unsure about my level of commitment and approach to innovation.',
                            'I would not be committed and would not focus on innovative approaches or market trends.'],
            'question_13': ['I am fascinated by technology and innovation.'],
            'question_14': ['Yes, I am fascinated by electrical engineering and the vast array of applications it offers in technology and industry'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },
              
                'Cyber Security and Mechanical': {
            'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
            'question_3': ['Yes, I am passionate about ensuring the safety and security of digital assets and information'],
            'question_4': ['Yes, I am passionate about cybersecurity.'],
              'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
         
            'question_7': ['','No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
            'question_9': ['Yes, I am fascinated by the principles of mechanics and engineering, and I enjoy designing and building machines'],
            'question_10': ['Extremely excited; it is my passion.'],
            'question_11': ['Business does not appeal to me.'],
            'question_12': ['I am unsure about my level of commitment and approach to innovation.',
                            'I would not be committed and would not focus on innovative approaches or market trends.'],
           'question_13': ['I am exploring different options.',
                            'I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],  'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },
            
           
          'BBA and English': {
            'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
           'question_5': ['Yes, I love playing with words and expressing my thoughts creatively'],
            'question_6': ['Literature and storytelling are my passions.'],
            'question_7': ['','No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
            'question_11': ['Absolutely, I am passionate about business.'],
            'question_12': ['I would be fully committed and eager to explore innovative approaches and market trends.'],
            'question_13': ['I am exploring different options.',
                            'I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        }, 
         
          'BBA and Chemistry': {
            'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
           'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
           'question_7': ['','Yes, I am passionate about chemistry and its role in shaping our understanding of the world'],
            'question_8': ['Absolutely, I find it incredibly intriguing.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
            'question_11': ['Absolutely, I am passionate about business.'],
            'question_12': ['I would be fully committed and eager to explore innovative approaches and market trends.'],
            'question_13': ['I am exploring different options.',
                            'I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        }, 
         
         'BBA and Electrical': {
            'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
           'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['','No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
            'question_11': ['Absolutely, I am passionate about business.'],
            'question_12': ['I would be fully committed and eager to explore innovative approaches and market trends.'],
            'question_13': ['I am fascinated by technology and innovation.'],
            'question_14': ['Yes, I am fascinated by electrical engineering and the vast array of applications it offers in technology and industry'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },  
           'BBA and Mechanical': {
            'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
            'question_2': ['I am not sure.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
           'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['','No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
             'question_9': ['Yes, I am fascinated by the principles of mechanics and engineering, and I enjoy designing and building machines'],
            'question_10': ['Extremely excited; it is my passion.'],
            'question_11': ['Absolutely, I am passionate about business.'],
            'question_12': ['I would be fully committed and eager to explore innovative approaches and market trends.'],
            'question_13': ['I am exploring different options.',
                            'I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },
              
        'English and Chemistry': {
             'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
          
            'question_2': ['I am not sure.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
            'question_5': ['Yes, I love playing with words and expressing my thoughts creatively'],
            'question_6': ['Literature and storytelling are my passions.'],
            'question_7': ['Yes, I am passionate about chemistry and its role in shaping our understanding of the world'],
            'question_8': ['Absolutely, I find it incredibly intriguing.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
            'question_11': ['Business does not appeal to me.'],
            'question_12': ['I am unsure about my level of commitment and approach to innovation.',
                            'I would not be committed and would not focus on innovative approaches or market trends.'],
            'question_13': ['I am exploring different options..',
                            'I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },
        
        'English and Electrical': {
             'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
          
            'question_2': ['I am not sure.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
            'question_5': ['Yes, I love playing with words and expressing my thoughts creatively'],
            'question_6': ['Literature and storytelling are my passions.'],
           'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
            'question_11': ['Business does not appeal to me.'],
            'question_12': ['I am unsure about my level of commitment and approach to innovation.',
                            'I would not be committed and would not focus on innovative approaches or market trends.'],
             'question_13': ['I am fascinated by technology and innovation.'],
            'question_14': ['Yes, I am fascinated by electrical engineering and the vast array of applications it offers in technology and industry'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },
        
        'English and Mechanical': {
             'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
          
            'question_2': ['I am not sure.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
            'question_5': ['Yes, I love playing with words and expressing my thoughts creatively'],
            'question_6': ['Literature and storytelling are my passions.'],
           'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
             'question_9': ['Yes, I am fascinated by the principles of mechanics and engineering, and I enjoy designing and building machines'],
            'question_10': ['Extremely excited; it is my passion.'],
            'question_11': ['Business does not appeal to me.'],
            'question_12': ['I am unsure about my level of commitment and approach to innovation.',
                            'I would not be committed and would not focus on innovative approaches or market trends.'],
            'question_13': ['I am exploring different options..',
                            'I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },
        
          'Chemistry and Electrical': {
             'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
          
            'question_2': ['I am not sure.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
             'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['Yes, I am passionate about chemistry and its role in shaping our understanding of the world'],
            'question_8': ['Absolutely, I find it incredibly intriguing.'],
            'question_9': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_10': ['Not excited at all.'],
            'question_11': ['Business does not appeal to me.'],
            'question_12': ['I am unsure about my level of commitment and approach to innovation.',
                            'I would not be committed and would not focus on innovative approaches or market trends.'],
             'question_13': ['I am fascinated by technology and innovation.'],
            'question_14': ['Yes, I am fascinated by electrical engineering and the vast array of applications it offers in technology and industry'],
          
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },
        
         'Chemistry and Mechanical': {
             'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
          
            'question_2': ['I am not sure.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
             'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['Yes, I am passionate about chemistry and its role in shaping our understanding of the world'],
            'question_8': ['Absolutely, I find it incredibly intriguing.'],
            'question_9': ['Yes, I am fascinated by the principles of mechanics and engineering, and I enjoy designing and building machines'],
            'question_10': ['Extremely excited; it is my passion.'],
            'question_11': ['Business does not appeal to me.'],
            'question_12': ['I am unsure about my level of commitment and approach to innovation.',
                            'I would not be committed and would not focus on innovative approaches or market trends.'],
             'question_13': ['I am exploring different options..',
                            'I am not sure.'],
            'question_14': ['I am not sure if this aligns with my career goals',
                                'No, I have other academic or career interests that Iam more inclined towards'],
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },
         
              'Electrical and Mechanical': {
             'question_1': ['No, I believe my independent efforts can create a more substantial and direct impact on projects'],
          
            'question_2': ['I am not sure.'],
            'question_3': ['I am not sure if this aligns with my career goals; I need more information about the cybersecurity field',
                               'No, I prefer a different career path that does not focus on cybersecurity'],
            'question_4': ['It is interesting, but not my primary focus',
                                'I am exploring this field but not fully committed',
                                'No, it does not excite me.'],
             'question_5': ['I am not sure if studying English in such depth is what I want, could you tell me more about the subjects and opportunities?',
                                'No, I am leaning towards studying something else that piques my interest'],
            'question_6': ['Literature does not interest me.'],
            'question_7': ['No, I have other academic or career interests that I am more inclined towards'],
            'question_8': ['No, it does not captivate my interest.'],
            'question_9': ['Yes, I am fascinated by the principles of mechanics and engineering, and I enjoy designing and building machines'],
            'question_10': ['Extremely excited; it is my passion.'],
            'question_11': ['Business does not appeal to me.'],
            'question_12': ['I am unsure about my level of commitment and approach to innovation.',
                            'I would not be committed and would not focus on innovative approaches or market trends.'],
        'question_13': ['I am fascinated by technology and innovation.'],
            'question_14': ['Yes, I am fascinated by electrical engineering and the vast array of applications it offers in technology and industry'],
          
            'question_15': ['High-demand job, regardless of passion',
                                'Passion-driven career, even if it is less in demand',
                                'A balance between demand and passion',
                                'Unsure about long-term career goals'],
            'question_16': ['I am not interested in aviation'],
            'question_17': ['I have not delved into aviation much yet.',
                            'Aviation does not captivate me.'],
              'question_18': ['Not particularly, it does not captivate me much. Although these are important principles, they do not significantly engage my interest', 'I am neutral about it. While these aspects are interesting, I am open to learning more about them to determine my level of fascination.'],
            'question_19': ['No, I do not see myself pursuing such a career.','I am unsure; it might interest me, but I need more information'],
            
        },
        
    }

    # Initialize a dictionary to store program scores
    program_scores = {}

    # Loop through the program criteria
    for program, criteria in program_criteria.items():
        score = 0  # Initialize the score for this program

        # Loop through the criteria for this program
        for question, valid_responses in criteria.items():
            user_response = user_responses.get(question, '')
            
            # Check if the user's response is in the list of valid responses
            if user_response in valid_responses:
                score += 1  # Increase the score for this program if criteria match

        program_scores[program] = score  # Store the score for this program

    # Find the program with the highest score (best match)
    recommended_program = max(program_scores, key=program_scores.get)

    return recommended_program
   


def success(request):
    return render(request, 'success.html')


def quinza(request):
    return render(request, 'quinza.html')

def programs(request):
         programs = Program.objects.all()
         return render(request, 'programs.html', {'programs': programs})
     
def images(request):
         images = Image.objects.all()
         return render(request, 'landing.html', {'images': images})     

def interest(request):
         programs = Program.objects.all()
         recommended_program = recommend_program(request.POST)
         return render(request, 'interest.html', {'recommended_program': recommended_program,'programs': programs}) 

def bert(request):
      universities = University.objects.all()
      return render(request, 'bert.html', {'universities': universities})
 
  

from django.shortcuts import render

# political_analysis/views.py
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from dataclasses import dataclass
import torch
import pandas as pd

# Define a data class to hold each data sample

@dataclass
class DataSample:  
    kamrankhan: str
    Label: int

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_samples, tokenizer, max_length):
        self.data_samples = data_samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        item = self.data_samples[index]
        encoding = self.tokenizer(item.kamrankhan, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        label = torch.tensor(item.Label, dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}
    
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
from io import BytesIO
import base64

    
import pandas as pd
import torch
import base64
from io import BytesIO
import plotly.graph_objs as go
from plotly.offline import plot
from django.shortcuts import render
from transformers import BertTokenizer, BertForSequenceClassification

def air_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='air.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['AIR']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['AIR'], sentiments)

        return render(request, 'political_analysis/air_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        return render(request, 'csv_not_found.html') 



    
def bahria_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='bahria.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['BAHRIA']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['BAHRIA'], sentiments)

        return render(request, 'political_analysis/bahria_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        return render(request, 'csv_not_found.html') 
    
        
def comsats_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='comsats.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['Comsats']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['Comsats'], sentiments)

        return render(request, 'political_analysis/comsats_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        return render(request, 'csv_not_found.html') 
    
        
def cust_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='cust.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['CUST']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['CUST'], sentiments)

        return render(request, 'political_analysis/cust_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        return render(request, 'csv_not_found.html') 
    
    
        
def gift_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='gift.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['GIFT']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['GIFT'], sentiments)

        return render(request, 'political_analysis/gift_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        return render(request, 'csv_not_found.html') 
    
        
def hitec_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='hitec.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['HITEC']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['HITEC'], sentiments)

        return render(request, 'political_analysis/hitec_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        return render(request, 'csv_not_found.html') 
    
        
def iiui_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='iiui.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['IIUI']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['IIUI'], sentiments)

        return render(request, 'political_analysis/iiui_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        return render(request, 'csv_not_found.html') 
    
        
def kust_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='kust.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['KUST']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['KUST'], sentiments)

        return render(request, 'political_analysis/kust_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        return render(request, 'csv_not_found.html') 
    
        
def lums_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='lums.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['LUMS']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['LUMS'], sentiments)

        return render(request, 'political_analysis/lums_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        return render(request, 'csv_not_found.html') 
    
        
def nust_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='nust.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['NUST']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['NUST'], sentiments)

        return render(request, 'political_analysis/nust_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        return render(request, 'csv_not_found.html') 
    
        

def pieas_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='pieas.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['PIEAS']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['PIEAS'], sentiments)

        return render(request, 'political_analysis/pieas_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        return render(request, 'csv_not_found.html') 

    
def quaid_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='quaid.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['Quaid e Azam']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['Quaid e Azam'], sentiments)

        return render(request, 'political_analysis/quaid_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        return render(request, 'csv_not_found.html') 
    

def riphah_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='riphah.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['RIPHAH']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['RIPHAH'], sentiments)

        return render(request, 'political_analysis/riphah_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        return render(request, 'csv_not_found.html') 
 

def uet_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='uet.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['Uet']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['Uet'], sentiments)

        return render(request, 'political_analysis/uet_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        return render(request, 'csv_not_found.html') 
 
 

def uniwah_political_review(request):
    # Load the pre-trained BERT model for sentiment analysis
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Read data from the CSV model
    data = CSVData.objects.filter(csv_name='uniwah.csv').first()

    if data:
        # Read CSV file using pandas
        df = pd.read_csv(data.file.path)

        # Perform sentiment analysis and store results
        sentiments = []
        for kamrankhan in df['Uniwah']:
            inputs = tokenizer.encode_plus(kamrankhan, add_special_tokens=True, return_tensors="pt")
            outputs = model(**inputs)
            sentiment = torch.argmax(outputs.logits).item()
            sentiments.append(sentiment)

        # Calculate sentiment counts
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Calculate the count of neutral sentiments

        # Create a bar chart for sentiment distribution
        labels = ['Negative', 'Neutral', 'Positive']
        counts = [negative_count, neutral_count, positive_count]

        data = [
            go.Bar(
                x=labels,
                y=counts,
                marker=dict(color=['red', 'purple', 'green']),
            )
        ]

        layout = go.Layout(
            title='Sentiment Analysis',
        )

        fig = go.Figure(data=data, layout=layout)

        # Save the plot as an HTML file
        plot_div = plot(fig, output_type='div')

        # Prepare data for rendering
        sentiment_data = zip(df['Uniwah'], sentiments)

        return render(request, 'political_analysis/uniwah_review.html', {
            'sentiment_data': sentiment_data,
            'plot_div': plot_div,  # Pass the Plotly plot as HTML
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
        })
    else:
        # Handle the case when the CSV file is not found
        
        return render(request, 'csv_not_found.html') 
    
    
# Import necessary libraries
import pandas as pd
from io import BytesIO
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import nltk.data
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# ... (Your other imports and code) ...

def cs(request):
    data = CSVData.objects.filter(csv_name='CS.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        # Define a custom color palette for the countplot
        custom_palette = {"pos": "green", "neg": "red"}

        for i in range(len(data)):
            X = data['CS'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['CS']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts with the custom color palette
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2, palette=custom_palette)
            plt.title('Graph')
            plt.xlabel('Prediction')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'cs.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')
    
    
def cyber(request):
    data = CSVData.objects.filter(csv_name='CYS.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        # Define a custom color palette for the countplot
        custom_palette = {"pos": "green", "neg": "red"}

        for i in range(len(data)):
            X = data['CYS'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['CYS']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts with the custom color palette
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2, palette=custom_palette)
            plt.title('Graph')
            plt.xlabel('Prediction')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'cyber.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')
    
def bba(request):
    data = CSVData.objects.filter(csv_name='BBA.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        # Define a custom color palette for the countplot
        custom_palette = {"pos": "green", "neg": "red"}

        for i in range(len(data)):
            X = data['BBA'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['BBA']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts with the custom color palette
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2, palette=custom_palette)
            plt.title('Graph')
            plt.xlabel('Prediction')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'bba.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')
    
def aviation(request):
    data = CSVData.objects.filter(csv_name='AVM.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        # Define a custom color palette for the countplot
        custom_palette = {"pos": "green", "neg": "red"}

        for i in range(len(data)):
            X = data['AVM'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['AVM']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts with the custom color palette
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2, palette=custom_palette)
            plt.title('Graph')
            plt.xlabel('Prediction')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'aviation.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')  


def mechanical(request):
    data = CSVData.objects.filter(csv_name='ME.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        # Define a custom color palette for the countplot
        custom_palette = {"pos": "green", "neg": "red"}

        for i in range(len(data)):
            X = data['ME'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['ME']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts with the custom color palette
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2, palette=custom_palette)
            plt.title('Graph')
            plt.xlabel('Prediction')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'mechanical.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')      
    
def electrical(request):
    data = CSVData.objects.filter(csv_name='EE.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        # Define a custom color palette for the countplot
        custom_palette = {"pos": "green", "neg": "red"}

        for i in range(len(data)):
            X = data['EE'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['EE']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts with the custom color palette
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2, palette=custom_palette)
            plt.title('Graph')
            plt.xlabel('Prediction')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'electrical.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')    
    
    
def english(request):
    data = CSVData.objects.filter(csv_name='English.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        # Define a custom color palette for the countplot
        custom_palette = {"pos": "green", "neg": "red"}

        for i in range(len(data)):
            X = data['English'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['English']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts with the custom color palette
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2, palette=custom_palette)
            plt.title('Graph')
            plt.xlabel('Prediction')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'english.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')    
      
      
def chemistry(request):
    data = CSVData.objects.filter(csv_name='Chemistry.csv')

    if data.exists():
        # Get the first object (assuming there is only one)
        csv_data = data.first()

        # Read the CSV file
        data = pd.read_csv(csv_data.file.path)

        # Initialize lists to store sentiment scores for each sentence
        pos_neg = list()

        # Load the English tokenizer from NLTK
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Load the spaCy model for English
        nlp = spacy.load('en_core_web_sm')

        sentiment_data = []
        overall_pos_count = 0
        overall_neg_count = 0

        # Define a custom color palette for the countplot
        custom_palette = {"pos": "green", "neg": "red"}

        for i in range(len(data)):
            X = data['Chemistry'][i]

            # Tokenize the text into sentences
            sents = tokenizer.tokenize(X)

            for sentence in sents:
                # Perform sentiment analysis on each sentence using NaiveBayesAnalyzer
                opinion = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
                pos_neg.append(opinion.sentiment[0])

                # Perform entity extraction using spaCy
                doc = nlp(sentence)
                set_list = []
                for ent in doc.ents:
                    set_list.append(''.join([ent.text, '-', ent.label_]))
                print("\nKey Entities\n..........\n" + str(set(set_list)))

                # Perform subject-verb-object extraction using the provided functions
                parse = nlp(sentence)
                print("\nSubjects\n..........\n" + str(findSVOs(parse)) + "\n\n********************************************************\n\n\n\n")

            # Calculate overall sentiment using the entire text
            overall_opinion = TextBlob(' '.join(data['Chemistry']), analyzer=NaiveBayesAnalyzer())
            sentiment = overall_opinion.sentiment

            # Calculate positive and negative counts
            if sentiment.classification == 'pos':
                overall_pos_count += 1
            elif sentiment.classification == 'neg':
                overall_neg_count += 1

            # Create a DataFrame to plot sentiment counts with the custom color palette
            df2 = pd.DataFrame(dict(x=pos_neg))
            ax = sns.countplot(x="x", data=df2, palette=custom_palette)
            plt.title('Graph')
            plt.xlabel('Prediction')
            plt.ylabel('Count')

            # Convert the plot to a base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            sentiment_graph = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

            # Store sentiment and graph in a dictionary for each item
            sentiment_data.append({
                'content': X,
                'sentiment': sentiment,
                'sentiment_graph': sentiment_graph,
            })

        # The rest of your code for graph generation, future prediction, and rendering

        overall_sentiment_count = {
            'positive_count': overall_pos_count,
            'negative_count': overall_neg_count,
        }

        return render(request, 'chemistry.html', {'data': sentiment_data, 'overall_sentiment_count': overall_sentiment_count})

    else:
        # Handle the case when 'csv2.csv' is not found in the database
        return render(request, 'csv_not_found.html')  
    
import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create the logs folder if it doesn't exist
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# political_analysis/views.py
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from dataclasses import dataclass
import torch
import pandas as pd
from transformers import TrainerCallback

# Define a data class to hold each data sample

@dataclass
class DataSample:  
    Review: str
    Label: int

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_samples, tokenizer, max_length):
        self.data_samples = data_samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        item = self.data_samples[index]
        encoding = self.tokenizer(item.Review, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        label = torch.tensor(item.Label, dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}

from transformers import Trainer, TrainingArguments
from django.conf import settings
import os
import logging
import pandas as pd
from django.shortcuts import render
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from transformers.trainer_callback import TrainerCallback

class LoggingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, args, state, control, **kwargs):
        logging.info(f"Starting epoch {state.epoch}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history:  # Check if log_history is not empty
            latest_log = state.log_history[-1]
            if 'train_loss' in latest_log:
                train_loss = latest_log['train_loss']
                logging.info(f"Step {state.global_step}: train_loss = {train_loss}")


def training(request):
    model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Read the single-sentence dataset from the CSV model
    training_data = CSVData.objects.filter(csv_name='career_path_data.csv').first()

    if training_data:
        training_df = pd.read_csv(training_data.file.path)

    # Prepare the data for training
    training_samples = [
        DataSample(Review=row['text'], Label=row['label']) for _, row in training_df.iterrows()
    ]

    train_dataset = CustomDataset(training_samples, tokenizer, max_length=128)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_total_limit=1,
        logging_dir='./logs',
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Configure logging
    log_folder = os.path.join(settings.BASE_DIR, 'logs')
    log_file_path = os.path.join(log_folder, 'fine_tuning.log')
    
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    logging.basicConfig(
        filename=log_file_path,
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
    )

    # Create the Trainer instance and start fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[LoggingCallback()],
    )

    # Fine-tune the model
    train_output = trainer.train()
    training_results = {
        'training_loss': train_output.metrics.get('train_loss', 'N/A'),
        'train_runtime': train_output.metrics.get('train_runtime', 'N/A'),
        'train_samples_per_second': train_output.metrics.get('train_samples_per_second', 'N/A'),
        'train_steps_per_second': train_output.metrics.get('train_steps_per_second', 'N/A'),
        'epoch': train_output.metrics.get('epoch', 'N/A'),
    }

    return render(request, 'training_result.html', {'training_results': training_results})
           

  
  
  
from django.shortcuts import render
from .models import CSVData

def get_prediction(request):
    if request.method == 'POST':
        text = request.POST.get('text')

        # Retrieve the prediction from the CSV based on the entered subject
        training_data = CSVData.objects.filter(csv_name='crime_data.csv').first()
        if training_data:
            training_df = pd.read_csv(training_data.file.path)
            label = training_df[training_df['text'] == text]['label'].values

            if len(label) > 0:
                label_value = label[0]
            else:
                label_value = "Text not found in CSV"

            return render(request, 'training_result.html', {'text': text, 'label': label_value})
    
    return render(request, 'training_form.html')
  

import csv
import re
import datetime
import dateutil.parser
from django.http import HttpResponse
from django.views import View
from textblob import TextBlob  # Import TextBlob for sentiment analysis

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'No Crime Indication'
    else:
        return 'Crime Indication'

def extract_information(sentence):
    # Regular expressions to extract date, location, time, and day information
    date_pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},\s\d{4}\b'
    location_pattern = r'(?:in|at)\s([A-Za-z\s-]+)\sDivision'
    time_pattern = r'at\s(\d{1,2}:\d{1,2}(?:\s(?:AM|PM))?)'
    day_pattern = r'on\s([A-Za-z\s]+),'
    month_pattern = r'(?:in|on)\s([A-Za-z\s]+),'
    crime_nature_pattern = r'a\s([A-Za-z\s]+)\stook place'
    
    date_match = re.search(date_pattern, sentence)
    location_match = re.search(location_pattern, sentence)
    time_match = re.search(time_pattern, sentence)
    day_match = re.search(day_pattern, sentence)
    month_match = re.search(month_pattern, sentence)
    crime_nature_match = re.search(crime_nature_pattern, sentence)
 
    date = None
    location = None
    time = None
    day = None
    month = None
    crime_nature = None

    if date_match:
        date = date_match.group(0)
        date = dateutil.parser.parse(date).strftime('%Y-%m-%d')
    if location_match:
        location = location_match.group(1)
    if time_match:
        time = time_match.group(1)
    if day_match:
        day = day_match.group(1)
    if month_match:
        month = month_match.group(1)
    if crime_nature_match:
        crime_nature = crime_nature_match.group(1)

    return date, location, time, day, month, crime_nature


def generate_career_data_view(request, *args, **kwargs):
    sentences = [
        # Your list of sentences here
"On October 15, 2023, in Lahore Division, at 9:00 PM, a robbery occurred at a local jewelry store.",
"In Sindh Division, on July 3, 2023, at 2:30 AM, a murder took place in Karachi.",
"On September 20, 2023, in Khyber Pakhtunkhwa Division, a kidnapping was reported in Peshawar at 11:45 AM.",
"In Balochistan Division, on May 8, 2023, at 7:15 PM, a case of car theft was reported in Quetta.",
"On January 12, 2023, in Gilgit-Baltistan Division, a burglary occurred at a residence in Skardu at 4:00 AM.",
"In Punjab Division, on March 5, 2023, at 1:20 PM, a drug trafficking operation was busted in Faisalabad.",
"On August 30, 2023, in Azad Jammu and Kashmir Division, a case of cybercrime was reported in Muzaffarabad at 3:30 PM.",
"In Islamabad Capital Territory, on June 18, 2023, at 10:45 AM, a white-collar fraud scheme was exposed.",
"On November 25, 2023, in the Federally Administered Tribal Areas (FATA), an assault took place in Khyber Agency at 6:30 PM.",
"In the Punjab Division, on April 9, 2023, a counterfeit currency operation was dismantled in Multan.",
"The weather in Lahore was pleasant on a sunny day in June.",
"People gathered at the park in Islamabad to celebrate a cultural festival.",
"The educational conference in Karachi was a great success, attracting experts from various fields.",
"A new healthcare facility was inaugurated in Quetta to serve the local community.",
"The cricket match between two local teams in Peshawar drew a large crowd of enthusiastic fans.",
    
    ]

    data = []

    for sentence in sentences:
        date, location, time, day, month, crime_nature = extract_information(sentence)
        sentiment = get_sentiment(sentence)
        data.append({
            'text': sentence,
            'label': sentiment,
            'date': date,
            'location': location,
            'time': time,
            'day': day,
            'month': month,
            'crime_nature': crime_nature,
        })

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="crime_data.csv"'

    csv_writer = csv.DictWriter(response, fieldnames=['text', 'label', 'date', 'location', 'time', 'day', 'month', 'crime_nature'])
    csv_writer.writeheader()
    csv_writer.writerows(data)

    return response



  
 