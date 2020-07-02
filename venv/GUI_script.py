from tkinter import *
import tkinter as tk
from selenium_down import downloadDataset
from threading import Timer
from regressionFunc import regFunc
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib import style
import os
plt.style.use('fivethirtyeight')

HEIGHT =700
WIDTH = 800
security_no = 0
train_graph= pd.DataFrame()
valid_graph = pd.DataFrame()

def onClickGP():
     global security_no
     security_no= entry.get()
     downloadDataset(security_no)
     genrate_excel = Timer(13.0,regFunc(security_no))

     from regressionFunc import train, valid, pred_price
     global train_graph,valid_graph
     train_graph = train.copy()
     valid_graph= valid.copy()
     generateGraphBtn['state'] = 'normal'
     button['state'] = 'disabled'
     entry['state'] = 'disabled'
     os.remove('/Users/lakshaymittal/Downloads/'+security_no+'.csv')

def onClickGG():
    global graph_df
    gen_Graph(train_graph,valid_graph)
    generateGraphBtn['state'] = 'disabled'


def gen_Graph(train, valid):

    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price INR ', fontsize=18)
    plt.plot(train['Close Price'])
    plt.plot(valid[['Close Price', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()



def reset():
    global security_no
    global valid_graph
    global train_graph
    security_no = 0
    train_graphf = pd.DataFrame()
    valid_graph = pd.DataFrame()
    generateGraphBtn['state'] = 'disabled'
    button['state'] = 'normal'
    entry['state'] = 'normal'
    entry.delete(0, END)




root = tk.Tk()

canvas= tk.Canvas(root, height =HEIGHT, width=WIDTH)
canvas.pack()

frame = tk.Frame(root, bg='#80c1ff')
frame.place(relx =0.1 , rely=0.1, relwidth=0.8, relheight=0.8)

label = tk.Label(frame, text='Rule Mining AI BOT' ,bg ='yellow' )
label.place(relwidth=1 , relheight=0.1)
button = tk.Button(frame, text='Get Predictions',bg='gray', command=onClickGP)
button.place(relx=0.6 ,rely=0.2)
entry = tk.Entry(frame)
entry.place(relx=0.2 ,rely=0.2)
entry.insert(0,"Enter BSE Security Number")
entry.configure(state='disabled')
generateGraphBtn=tk.Button(frame, text='Generate Graph',bg='gray',state='disabled', command=onClickGG)
generateGraphBtn.place(relx=0.6 ,rely=0.3)
resetBtn=tk.Button(frame, text='Reset',bg='gray',command=reset)
resetBtn.place(relx=0.5, rely=0.5)



def on_click(event):
      entry.configure(state=NORMAL)
      entry.delete(0, END)

    # make the callback only work once
      entry.unbind('<Button-1>', on_click_id)

on_click_id = entry.bind('<Button-1>', on_click)







root.mainloop()