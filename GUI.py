from tkinter import *
import numpy  as np
import matplotlib.pyplot as plt
import main as mlp
class GUI:
    def __init__(self, master):
        self.master = master
        master.title("NN-MLP")
        self.layers = []
        self.alpha = 0
        self.epoc = 0
        self.dset = 0

        frame=Frame(master, width=550, height=350)
        frame.pack()

        self.label = Label(master, text="Mnist = 1\nCar = 2\nIris = 3\n", anchor = "w")
        self.label.place(x=30, y=200, height=90, width=100)
        
        self.entry1 = Entry(master)
        self.entry1.place(x=10, y=90, height=30, width=100)#layer
        #self.entry2 = Entry(master)
        #self.entry2.place(x=10, y=90, height=30, width=100)#alpha
        #self.entry3 = Entry(master)
        #self.entry3.place(x=10, y=130, height=30, width=100)#LR
        self.entry4 = Entry(master)
        self.entry4.place(x=10, y=130, height=30, width=100)#Dataset
        self.entry5 = Entry(master)
        self.entry5.place(x=10, y=170, height=30, width=100)#epochs

        self.text_1 = Text(master)
        self.text_1.place(x=250, y=50, height=150, width=250)
#Add layer
        self.Layer_button = Button(master, text="Add Layer", command=self.add_layer)
        self.Layer_button.place(x=120, y=90, height=30, width=100)
#reg
        #self.alpha_button = Button(text="Reg Paremater", command=self.add_alpha)
        #self.alpha_button.place(x=120, y=90, height=30, width=100)
#Learning rate
        #self.lr_button = Button(master, text="Learning Rate", command=self.test)
        #self.lr_button.place(x=120, y=130, height=30, width=100)
#dataset      
        self.dataset_button = Button(master, text="DataSet'", command=self.add_dataset)
        self.dataset_button.place(x=120, y=130, height=30, width=100)
#Epocs
        self.epochs_button = Button(master, text="Epochs", command=self.add_epoc)
        self.epochs_button.place(x=120, y=170, height=30, width=100)
#run       
        self.run_button = Button(master, text="RUN", command=self.start)
        self.run_button.place(x=150, y=250, height=30, width=100)


    def start(self):
        self.text_1.delete('1.0', END)
        self.text_1.insert(END,"Starting...\n")
        acc = mlp.initialize(self.layers,self.epoc,self.dset)
        self.text_1.insert(END,"Done. \nAccuracy:" + acc)
        
    def add_layer(self):
        self.text_1.delete('1.0', END)
        entry = self.entry1.get()
        self.layers.append(int(entry))
        self.text_1.insert(END,"Added layer.\nCurrent Hidden Layer:" + ','.join([str(x) for x in self.layers]))

    def add_alpha(self):
        self.text_1.delete('1.0', END)
        entry = self.entry2.get()
        self.alpha = float(entry)
        self.text_1.insert(END,"Set Regilarization Parameter:" + str(self.alpha))

    def add_epoc(self):
        self.text_1.delete('1.0', END)
        entry = self.entry5.get()
        self.epoc = int(entry)
        self.text_1.insert(END,"Set Epocs To:" + str(self.epoc))

    def add_dataset(self):
        self.text_1.delete('1.0', END)
        entry = self.entry4.get()
        self.dset = int(entry)
        self.text_1.insert(END,"Chose DataSet:" + str(self.dset))
root = Tk()
my_gui = GUI(root)
root.mainloop()
