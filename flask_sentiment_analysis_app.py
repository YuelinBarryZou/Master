from flask import Flask, render_template, request, session, send_file, Response
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.io.json import json_normalize
import csv
import os
import json #Turn json to dict
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator #wordcloud added after 7/29
import matplotlib.pylab as plt
from PIL import Image

import string
from io import BytesIO
import plotly
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob

# Sentiment analysis function using VADER
def vader_sentiment_scores(data_frame):
    # Define SentimentIntensityAnalyzer object of VADER.
    SID_obj = SentimentIntensityAnalyzer()
    # Added Lexion to adjust the review score
    new_lexicon = {
    'incorrect':-3.4,
    'inaccurate':-3.2,
    'clickbait':-2,
    'ads':-1.8,
    'zero':-3,
    '0':-3,
    'yet':-2.3,
    'not':-2.8,
    'loading':-2.2,
    'crashed':-2.6,
    'slow':-2.3,
    'uninstalled':-4,
    'slow':-1.2,
    'long':-3,
    'used':-2.4,
    'terrible':-3,
    'before':-2,
    'but':-3,
    'BUT':-3,
    'accurate':2,
    'were':-2.1,
    'paid':-1.2,
    'pay':-1.3,
    'because':-1.2,
    'downhill':-1.8,
    'intrusive':-2.9,
    'cant':-1.5,
    "can't":-1.5,
    'cannot':-1.5
}
    SID_obj.lexicon.update(new_lexicon)
    # calculate polarity scores which gives a sentiment dictionary,
    # Contains pos, neg, neu, and compound scores.
    sentiment_list = []
    #add a threshold
    

    for row_num in range(len(data_frame)):
        sentence = data_frame['Review Text'][row_num]
        
        if sentence == None:
            sentiment_list.append('N/A')
            continue
        polarity_dict = SID_obj.polarity_scores(sentence)
 
        # Calculate overall sentiment by compound score
        if polarity_dict['compound'] >= 0.05:
            sentiment_list.append("Positive")
 
        elif polarity_dict['compound'] <= - 0.05:
            sentiment_list.append("Negative")
 
        else:
            sentiment_list.append("Neutral")
 
    data_frame['Sentiment'] = sentiment_list
 
    return data_frame
 
 
#*** Backend operation
# Read comment csv data
# df = pd.read_csv('data/comment.csv')
 
# WSGI Application
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name
app = Flask(__name__, template_folder='templates')
 
app.secret_key = 'Yuelin'
 
#store the none type csv first
app.current_file_json = None
# @app.route('/')
# def welcome():
#     return "Ths is the home page of Flask Application"
 
@app.route('/')
def index():
    return render_template('index_upload_and_show_data.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        uploaded_file = request.files['uploaded-file']
        df = pd.read_csv(uploaded_file)
        #retrieve full ratings
        full_ratings = df[['Star Rating', 'Review Last Update Date and Time']]
        #Drop the NaN Text Review
        #df = df[df['Review Text'].notna()]

        #session['uploaded_csv_file'] = df.to_json()
        #store cookie
        app.current_file_json=df.to_json()
        
        return render_template('index_upload_and_show_data_page2.html')
 
@app.route('/show_data')
def showData():
    # Get uploaded csv file from session as a json value
    #uploaded_json = session.get('uploaded_csv_file', None)
    uploaded_json =app.current_file_json
    if uploaded_json == None:
        return render_template('index_upload_and_show_data_page3.html')
    
    # Convert json to data frame
    #print(json.loads(uploaded_json)) # Debug
    uploaded_df = pd.DataFrame.from_dict(json.loads(uploaded_json))
    # Convert dataframe to html format
    uploaded_df=uploaded_df.reset_index(drop=True)
    uploaded_df = uploaded_df.head(5)
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_data.html', data=uploaded_df_html)
 
@app.route('/sentiment')
def SentimentAnalysis():
    # Get uploaded csv file from session as a json value
    #uploaded_json = session.get('uploaded_csv_file', None)
    uploaded_json=app.current_file_json
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(json.loads(uploaded_json))
    # Apply sentiment function to get sentiment score
    uploaded_df_sentiment = vader_sentiment_scores(uploaded_df)
    # Show percentage of the review in POS/NEG/NET
    sentiment=pd.DataFrame() 
    total=uploaded_df_sentiment['Sentiment'].count()
    sentiment['Positive']=[f"{round(100 * uploaded_df_sentiment['Sentiment'].value_counts()['Positive'] / total,2)}%" ] 
    sentiment['Negative']=[f"{round(100 * uploaded_df_sentiment['Sentiment'].value_counts()['Negative'] / total,2)}%" ]
    sentiment['Neutral']=[f"{round(100 * uploaded_df_sentiment['Sentiment'].value_counts()['Neutral'] / total,2)}%" ]
    #
    
    #
    sentiment_df_html =sentiment.to_html()
    uploaded_df_html = uploaded_df_sentiment.to_html()

   

    #Star rating dataframe
    ratings=uploaded_df['Star Rating'].value_counts()
    #Turn to dataframe load its count
    ratings=ratings.to_frame().reset_index()
    ratings.columns = ['Ratings', 'Counts']
    ratings=ratings.sort_values('Ratings', ascending=False)
    #ratings['Percentage']=ratings['Counts'] /
    ratings_df=ratings.style.hide_index()
    
    rating_df_html=ratings_df.to_html()
    return render_template('show_data.html', data=uploaded_df_html,sentiment=sentiment_df_html,rating=rating_df_html)
    


@app.route('/dashboards')
def dashData():
    
    #load data
    uploaded_json=app.current_file_json
   
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(json.loads(uploaded_json))
    
    # Apply sentiment function to get sentiment score
    uploaded_df_sentiment = vader_sentiment_scores(uploaded_df)
    textReview=uploaded_df_sentiment[uploaded_df_sentiment['Review Text'].notna()]
    uSReview=textReview[textReview['Location'] == 'US']
    textReview=textReview['Review Text'].to_frame().reset_index()
    
    sentiment=vader_sentiment_scores(textReview)
    #Store positive and negative review
    neg=sentiment[sentiment['Sentiment']=='Negative']
    #pos=sentiment[sentiment['Sentiment']=='Positive']
    neg_review=neg['Review Text']




    neg_counts=neg_review.value_counts().to_frame().reset_index()
    
    neg_counts.columns = ['Review', 'Counts']
    top5_neg=neg_counts.head(5)
    top5_neg=top5_neg.sort_values(by='Counts',ascending=True)
    
    



    #New Code: with the Words Frequent update
    
    app_str='\n'.join(textReview['Review Text'])
    app_str=app_str.lower()
    app_str_rv = app_str.split()
    table = str.maketrans("","",string.punctuation)
    stripped = [w.translate(table) for w in app_str_rv]
    
    #sent = app_str
    pos_word_list = []
    neg_word_list = []
    neu_word_list = []

    for word in stripped:
        testimonial = TextBlob(word)
        if testimonial.sentiment.polarity >= 0.5:
            pos_word_list.append(word)
        elif testimonial.sentiment.polarity <= -0.5:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)
    #Negative
    neg_word_df=pd.DataFrame(neg_word_list,columns=(['NegativeWord']))
    neg_word_df=pd.DataFrame(neg_word_df.value_counts().head(10),columns=(['Counts'])).reset_index()
    neg_word_df=neg_word_df.sort_values('Counts',ascending=True)
    #positive 
    pos_word_df=pd.DataFrame(pos_word_list,columns=(['PositiveWord']))
    pos_word_df=pd.DataFrame(pos_word_df.value_counts().head(10),columns=(['Counts'])).reset_index()
    pos_word_df=pos_word_df.sort_values('Counts',ascending=True)

    fig=px.bar(neg_word_df, x='Counts',y='NegativeWord', title="Most Frequent Negative Words")
    fig.update_layout(barmode='stack')


    fig2=px.bar(pos_word_df, x='Counts',y='PositiveWord',title="Most Frequent Positive Words")
    fig2.update_layout(barmode='stack')

    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    posgraphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    #New Code2: Sentiment Percentage Dashboard Update
    
    total=uploaded_df_sentiment['Sentiment'].count()
    

    pos=uploaded_df_sentiment['Sentiment'].value_counts()['Positive']
    neg=uploaded_df_sentiment['Sentiment'].value_counts()['Negative']
    neu=uploaded_df_sentiment['Sentiment'].value_counts()['Neutral']



    labels = ['Positive','Negative','Neutral']
    values = [pos,neg,neu]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)],layout=go.Layout(
        title=go.layout.Title(text="Review Sentiment Percentage")))

    PercentageJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    

    ## Table Start ##
    #Description of the most serious Neg comment
    
    
    review_data=uSReview.reset_index()#.drop(columns=['index','Unnamed: 0','Unnamed: 0.1','Package Name','Review Link'])
    sentiment_data=vader_sentiment_scores(review_data)
    analyzer = SentimentIntensityAnalyzer()
    sentiment_data['compound'] = [analyzer.polarity_scores(x)['compound'] for x in sentiment_data['Review Text']]
    sentiment_data['neg'] = [analyzer.polarity_scores(x)['neg'] for x in sentiment_data['Review Text']]
    sentiment_data['pos'] = [analyzer.polarity_scores(x)['pos'] for x in sentiment_data['Review Text']]
    top_negative=sentiment_data.sort_values('compound',ascending=True).head(20)
    top_positive=sentiment_data.sort_values('compound',ascending=False).head(20)
    #Filter out the US only comment
    top_negative_us=top_negative[top_negative['Location'] == 'US']
    top_positive_us=top_positive[top_positive['Location'] == 'US']
    wc_neg=top_negative_us['Review Text'].to_frame().reset_index().drop(columns=['index'])
    wc_pos=top_positive_us['Review Text'].to_frame().reset_index().drop(columns=['index'])


    wc_neg=wc_neg['Review Text']
    wc_pos=wc_pos['Review Text']





    #Table format
    values = [wc_neg]

    fig = go.Figure(data=[go.Table(
    columnorder = [1,2],
    columnwidth = [80,400],
    header = dict(
        values = [
                    ['<b>DESCRIPTION: Top Negative Comment(DESC Order)</b>']],
        line_color='darkslategray',
        fill_color='royalblue',
        align=['left','center'],
        font=dict(color='white', size=12),
        height=40
    ),
    cells=dict(
        values=values,
        line_color='darkslategray',
        fill=dict(color=['paleturquoise', 'white']),
        align=['left', 'center'],
        font_size=12,
        height=30)
        )
    ])
    

    DescriptionJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


    #pos chart
    values = [wc_pos]

    fig = go.Figure(data=[go.Table(
    columnorder = [1,2],
    columnwidth = [80,400],
    header = dict(
        values = [
                    ['<b>DESCRIPTION: Top Positive Comment(DESC Order)</b>']],
        line_color='darkslategray',
        fill_color='orange',
        align=['left','center'],
        font=dict(color='white', size=12),
        height=40
    ),
    cells=dict(
        values=values,
        line_color='darkslategray',
        fill=dict(color=['paleturquoise', 'white']),
        align=['left', 'center'],
        font_size=12,
        height=30)
        )
    ])
    

    PosDescriptionJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    full_ratings=uploaded_df[['Star Rating', 'Review Last Update Date and Time']]

    full_ratings['DateTime']=pd.to_datetime(full_ratings['Review Last Update Date and Time'])
    full_ratings['Month_Year']=full_ratings['DateTime'].dt.to_period('M')
    #sentiment_data['Month_Year']=json.dumps(sentiment_data['Month_Year'], sort_keys=True, default=str)
    full_ratings['Month_Year']=full_ratings['Month_Year'].dt.strftime('%Y-%m')
    

    full_ratings = full_ratings.sort_values(by="Month_Year")
    #fig=px.line(sentiment_data, x='Month_Year',y='Star Rating', title="YoY of Stars")
    #fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'})
    months = full_ratings['Month_Year'].unique()


    ratings_byM=full_ratings.groupby('Month_Year')['Star Rating'].value_counts()
    ratings_byM=dict(ratings_byM)

    ratings = full_ratings['Star Rating']

    fig = go.Figure()
    for i in sorted(full_ratings['Star Rating'].unique()):
        counts=[]
        for j in sorted(months):
            try:
                counts.append(ratings_byM[(j,i,)])
            except KeyError:
                continue
            
        fig.add_trace(go.Bar(
            x=months,
            y=counts,
            name=f'{i} Star',
            
        ))



    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='group', xaxis_tickangle=-45,title='Ratings By Month')

    barRatingJson = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)



    return render_template('dashboards.html',graphJSON=graphJSON, posgraphJSON = posgraphJSON,PercentageJSON = PercentageJSON,PosDescriptionJSON = PosDescriptionJSON,DescriptionJSON =DescriptionJSON, barRatingJson = barRatingJson)
    #return send_file(img, mimetype='image/png')

@app.route('/review_pie.png')
def reviewPie():
    uploaded_json=app.current_file_json
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(json.loads(uploaded_json))

    ratings=uploaded_df['Star Rating'].value_counts()
    #Turn to dataframe load its count
    ratings=ratings.to_frame().reset_index()
    ratings.columns = ['Ratings', 'Counts']
    ratings=ratings.sort_values('Ratings', ascending=False)
    #ratings['Percentage']=ratings['Counts'] /
    ratings_df=ratings.style.hide_index()
    #get the content of review
    rt_list = ratings['Ratings'].tolist()
    ct_list = ratings['Counts'].tolist()
    
    y = np.array(ct_list)
    mylabels = rt_list
    
    #Plot the pie chart
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    myexplode = [0.2, 0,0,0,0]
    axis.set_title('Ratings Breakdown')
    axis.pie(y, labels = mylabels, shadow=True,explode=myexplode,autopct='%1.2f%%')
    
    axis.legend()
    
    output = BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
    
   
    




@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    
    uploaded_json=app.current_file_json
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(json.loads(uploaded_json))
    # Apply sentiment function to get sentiment score
    uploaded_df_sentiment = vader_sentiment_scores(uploaded_df)
    textReview=uploaded_df_sentiment[uploaded_df_sentiment['Review Text'].notna()]
    textReview=textReview['Review Text'].to_frame().reset_index()
    
    #pos_review=pos['Review Text']
    
    app_str='\n'.join(textReview['Review Text'])

    sent = app_str
    pos_word_list = []
    neg_word_list = []
    neu_word_list = []

    for word in sent.split():
        testimonial = TextBlob(word)
        if testimonial.sentiment.polarity >= 0.5:
            pos_word_list.append(word)
        elif testimonial.sentiment.polarity <= -0.5:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)
    #word clouds
    #Stop words
    
    stopwords=set(STOPWORDS)
    stopwords.add("weather")
    stopwords.add("now")
    stopwords.add("app")
    stopwords.add('time')
    stopwords.add('s')
    stopwords.add('t')
    #stopwords.add('Accuweather')
    #Load accuweather logo


    mask= np.array(Image.open(r'/Users/barry/nlp_flask/accuwea.png'))
    
    #text=df[df['Review Text'].notna()]
    wordcloud = WordCloud(stopwords = stopwords,width=1600,height=800 ,colormap='Set1', collocations=False,mask=mask,background_color="white").generate(''.join(neg_word_list))
    fig = Figure() #plt.figure(figsize=(20,10),facecolor='k')
    #plt.imshow(wordcloud,interpolation='bilinear')
    #plt.axis('off')
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title('Negative Comments WordClouds')
    axis.imshow(wordcloud,interpolation='bilinear')
    axis.axis('off')
    #plt.tight_layout(pad=0)
    #plt.show()
  
    return fig

if __name__=='__main__':
    app.run(debug = True)