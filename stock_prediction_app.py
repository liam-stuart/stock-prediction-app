import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNNCell,Conv1D,Flatten
from sklearn.metrics import mean_absolute_error
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import flask
from data_collection import extract_symbols
import warnings
warnings.filterwarnings('ignore')

headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0'}

#After defining the symbols and names, we need to filter out the ones we previously removed from the web scraping file.
def check_symbols(symbols):
    data=[]
    directory=os.getcwd()
    subfolders=[x[0] for x in os.walk(directory+'\\stock_data')][1:]
    for subfolder in subfolders:
        data.append(subfolder.split('\\')[-1].split()[0])
    return [i for i in symbols if i in data] 

#After creating a dict with the symbols and names, filter the names to only include valid entries.
def name_dict(symbols,stock_dict):
    dict={}
    for symbol in symbols:
        dict[symbol]=stock_dict[symbol]
    return dict

#Extract the currency information from the data for display later.
def get_currencies(symbols):
    currencies=[0]*len(symbols)
    directory=os.getcwd()
    subfolders=[x[0] for x in os.walk(directory+'\\stock_data')][1:]
    for subfolder in subfolders:
        if subfolder.split('\\')[-1].split()[0] in symbols:
            i=symbols.index(subfolder.split('\\')[-1].split()[0])
            currency=subfolder.split('\\')[-1].split()[1]
            currencies[i]=currency
    return currencies

#Given a dataframe, perform a 70-20-10 train-validation-test split. Important to not do this randomly with train_test_split due to the time dependant ordering of the data.
#Also retaining minimum and maximum as we will scale the data for training and then undo for display purposes.
def train_val_test(df):
    n=len(df)
    train=df[0:int(n*0.7)]
    val=df[int(n*0.7):int(n*0.9)]
    test=df[int(n*0.9):]
    min=train['Adj. Close'].min()
    max=train['Adj. Close'].max()
    return train,val,test,min,max

#Windowing the data where we use a week's worth of data to predict the next day.
def create_windows(df,window_length=7):
    windows=[]
    labels=[]
    X=df['Adj. Close'].to_numpy()
    n=len(X)
    for i in range(n-window_length):
        window=X[i:i+window_length]
        label=X[i+window_length]
        windows.append(window)
        labels.append(label)
    windows,labels=tf.convert_to_tensor(windows),tf.convert_to_tensor(labels)
    windows=tf.reshape(windows,(windows.shape[0],windows.shape[1],1))
    return windows,labels

#Helper function to generate windows on train,validation and test sets.
def windows(train,val,test):
    train_windows,train_labels=create_windows(train)
    val_windows,val_labels=create_windows(val)
    test_windows,test_labels=create_windows(test)
    return train_windows,train_labels,val_windows,val_labels,test_windows,test_labels

#Training loop for the models. We rescale the MAEs and predictions for tomorrow to account for the initial scaling we perform.
def train(model,df):
    train,val,test,min,max=train_val_test(df)
    train['Adj. Close']=(train['Adj. Close']-min)/(max-min)
    val['Adj. Close']=(val['Adj. Close']-min)/(max-min)
    test['Adj. Close']=(test['Adj. Close']-min)/(max-min)
    train_windows,train_labels,val_windows,val_labels,test_windows,test_labels=windows(train,val,test)

    model.fit(train_windows,train_labels,epochs=10,verbose=False)
    val_preds=model.predict(val_windows,verbose=False)
    test_preds=model.predict(test_windows,verbose=False)
    mae_val=mean_absolute_error(val_preds,val_labels)*(max-min)
    mae_test=mean_absolute_error(test_preds,test_labels)*(max-min)
    tomorrow=tf.convert_to_tensor(test['Adj. Close'][-7:])
    tomorrow=tf.reshape(tomorrow,(1,7,1))
    tomorrow_pred=model.predict(tomorrow,verbose=False)*(max-min)+min
    return mae_val,mae_test,tomorrow_pred[0][0]





#Displays a graph of the full dataset vs. the model's predictions across the dataset.
def create_plot(model,df,symbol):
    _,_,_,min,max=train_val_test(df)
    df['Adj. Close']=(df['Adj. Close']-min)/(max-min)
    windows,labels=create_windows(df)
    data=np.concatenate(([[None] for i in range(7)],model.predict(windows,verbose=False)),axis=0)
    df[['Adj. Close Pred']]=data
    df['Adj. Close']=df['Adj. Close']*(max-min)+min
    df['Adj. Close Pred']=df['Adj. Close Pred']*(max-min)+min
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj. Close'],
                    mode='lines',name='<b>Actual</b>'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj. Close Pred'],
                    mode='lines',name='<b>Predicted</b>'))
    fig.update_layout(
    title={'text':'<b>Plot of {} Stock Price vs. Model Predictions </b>'.format(symbol)+' <b>{}</b>'.format(currency_dict[symbol]),'x':0.5},
    font={'size':25,'family':'Times New Roman','color':'black'},
    xaxis_title='<b>Date</b>',
    yaxis_title='<b>Adjusted Closing Price</b>',
    height=750
    )
    return fig

#Defining the models.
model1= Sequential([
Conv1D(filters=256,kernel_size=5,padding='same',activation='relu'),
Flatten(),
Dense(1)
])

model2 = Sequential([
LSTM(256),
Dense(1)
])

model3 = Sequential([
GRU(256),
Dense(1)
])

#Construct the relevant dictionaries, also define some text styles for different parts of the dashboard.
names,symbols=extract_symbols()
stock_dict=dict(zip(symbols,names))
symbols=check_symbols(symbols)
currencies=get_currencies(symbols)

stock_dict=name_dict(symbols,stock_dict)
currency_dict=dict(zip(names,currencies))
model_dict={'1D Convolutional':model1,'LSTM':model2,'GRU':model3}

style_text={'font-size':30,'display':'inline','margin-left':'40px'}
style_text2={'font-size':30,'display':'inline'}
style_dropdown={'font-size':30,'width':'30%','margin-left':'40px'}

#Constructing the dashboard.
server = flask.Flask(__name__)
app = dash.Dash(__name__,server=server,external_stylesheets=[dbc.themes.CYBORG])

app.layout = html.Div([
    html.Br(),
    #Title
    html.H1('Stock Price Prediction Dashboard',style={'textAlign':'center','font-size':48}),
    #Stock dropdown
    html.Div([
            html.B(
                'Select the stock price you would like predictions for:',
                style=style_text
            ),
        ]),
    html.Div([dcc.Dropdown(
            id='dropdown-stock',
            options=[{'label': stock_dict[stock],'value':stock} for stock in stock_dict],
            value=symbols[0],
            clearable=False
        )],
        style=style_dropdown
        ),
    html.Br(),
    #Year dropdown
    html.Div([
            html.B(
                'Select the number of years of data you would like to use for training:',
                style=style_text
            )
        ]),
    html.Div([
        dcc.Dropdown(
            id='dropdown-year',
            options=[{'label': '1 Year', 'value': '1Y'},
                  {'label': '3 Years', 'value': '3Y'},
                  {'label': '5 Years', 'value': '5Y'},],
            value='1Y',
            clearable=False,
        )],
        style=style_dropdown
        ),
    html.Br(),
    #Model dropdown
    html.Div([
            html.B(
                'Select the model architecture you would like to use to train on the data:',
                style=style_text
            )
        ]),
    html.Div([
        dcc.Dropdown(
            id='dropdown-model',
            options=[{'label': '1D Convolutional', 'value': '1D Convolutional'},
                    {'label': 'LSTM', 'value': 'LSTM'},
                    {'label': 'GRU', 'value': 'GRU'}],
            value='1D Convolutional',
            clearable=False,
        )],
        style=style_dropdown
        ),
    html.Br(),
    html.Br(),
    html.Div(id='loading-message',style={'font-size':30}),
    html.Br(),
    dcc.Loading(id='loading_output',children=[html.Div(id='output-container')],type='default')

])

#Shows a loading message while the models are training, does not show when app is first loaded due to duplicate outputs in callback functions.
@app.callback(
    Output(component_id='loading-message', component_property='children'),
    Output(component_id='loading-message', component_property='style',allow_duplicate=True),
    Input(component_id='dropdown-stock',component_property='value'),
    Input(component_id='dropdown-year',component_property='value'),
    Input(component_id='dropdown-model',component_property='value'),
    prevent_initial_call=True)

def loading_message(stock,year,model):
    return html.B('Training model, please wait...'),{'font-size':30,'display':'block','margin-left':'40px'}

#Given a stock name, number of years, and model type: train the model, hide the loading message and display the results.
@app.callback(
    Output(component_id='loading-message', component_property='style'),
    Output(component_id='output-container', component_property='children'),
    Input(component_id='dropdown-stock',component_property='value'),
    Input(component_id='dropdown-year',component_property='value'),
    Input(component_id='dropdown-model',component_property='value'))

def model_results(stock,year,model):
    df=pd.read_csv('stock_data\\'+stock+' {}'.format(currency_dict[stock_dict[stock]])+'\\'+year).copy()
    model_copy=tf.keras.models.clone_model(model_dict[model])
    model_copy.compile(optimizer='adam',loss='mae')
    mae_val,mae_test,tomorrow_pred=train(model_copy,df)
    fig=create_plot(model_copy,df,stock_dict[stock])
    val_error=html.Div([html.B('Validation Set Mean Average Error:',style=style_text), html.P(' {:.5f}'.format(mae_val),style=style_text2)])
    test_error=html.Div([html.B('Test Set Mean Average Error:',style=style_text), html.P(' {:.5f}'.format(mae_test),style=style_text2)])
    tomorrow=html.Div([html.B('Prediction for tomorrow\'s stock price based on the past 7 days:',style=style_text),html.P(' {:.2f}'.format(tomorrow_pred),style=style_text2)])
    return {'display':'none'},(val_error,html.Br(),test_error,html.Br(),tomorrow,html.Br(),html.Div([dcc.Graph(figure=fig)]))


#Run the app.
if __name__ == '__main__':
    app.run_server(debug=False)
