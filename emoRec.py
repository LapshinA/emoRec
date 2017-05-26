import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import pyaudio
import wave
import threading
from threading import Lock
 
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.externals import joblib
from numpy import float64

import csv as cs
import os
import time
import subprocess
import ntpath
import glob
import re
from string import digits

mins = pd.Series.from_csv('Models/BigDataSet/mins.csv', sep=';', header=None)
maxs = pd.Series.from_csv('Models/BigDataSet/maxs.csv', sep=';', header=None)
means = pd.Series.from_csv('Models/BigDataSet/means.csv', sep=';', header=None)
stds = pd.Series.from_csv('Models/BigDataSet/stds.csv', sep=';', header=None)

class PlayAudioFile:
    chunk = 1024

    def __init__(self, file):
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )

    def play(self):
        data = self.wf.readframes(self.chunk)
        while data != '':
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)

    def close(self):
        self.stream.close()
        self.p.terminate()

class w_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("emoRec")
        Form.setWindowTitle("emoRec")
        Form.resize(260, 380)

        self.pushButton = QPushButton(Form)
        self.pushButton.move(30,340)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Start")

        self.stopButton = QPushButton(Form)
        self.stopButton.move(145,340)
        self.stopButton.setObjectName("stopButton")
        self.stopButton.setText("Play")
        
        self.radio1 = QRadioButton(Form)
        self.radio1.setText("Mic")
        self.radio1.setObjectName("radio1")

        self.radio2 = QRadioButton(Form)
        self.radio2.setText("Wav")
        self.radio2.setObjectName("radio2")
        
        self.radio3 = QRadioButton(Form)
        self.radio3.setText("Long Wav")
        self.radio3.setObjectName("radio3")

        
        self.radio1.toggled.connect(self.fradio1)
        self.radio1.setChecked(True)
        self.radio2.toggled.connect(self.fradio2)
        self.radio3.toggled.connect(self.fradio3)

        self.vLayout = QVBoxLayout()
        self.vLayout.addWidget(self.radio1)
        self.vLayout.addWidget(self.radio2)
        self.vLayout.addWidget(self.radio3)
        
        self.groupBox = QGroupBox(Form)
        self.groupBox.setTitle("Mode")
        self.groupBox.setLayout(self.vLayout)
        self.groupBox.move(20,20)

        self.dataSetCBox = QComboBox(Form)
        self.dataSetCBox.addItems(['Big dataset','Small dataset'])
        self.dataSetCBox.resize(220,30)
        self.dataSetCBox.move(20,130)

        self.dataSetCBox.currentIndexChanged.connect(self.slotComboDataSet)
        
        self.modelCBox = QComboBox(Form)
        self.modelCBox.addItems(['Naive Bayes','Linear SVM','Polinomial SVM','Radial SVM','Sigmoid SVM'])
        self.modelCBox.resize(220,30)
        self.modelCBox.move(20,165)
        self.modelCBox.currentIndexChanged.connect(self.slotComboAlgorithm)
        self.modelCBox.activated.connect(self.slotComboAlgorithm)
        
        self.fileDialog = QFileDialog()
        
        self.resText = QTextEdit()
        self.resText.resize(60,60)
        self.resText.move(150,20)

        self.resTable = QTableWidget(Form)
        self.resTable.setRowCount(4);
        self.resTable.setColumnCount(2);
        self.resTable.setHorizontalHeaderLabels(['Emotion','Probability'])
        self.resTable.verticalHeader().setDefaultSectionSize(20);

        self.resTable.setItem(0, 0, QTableWidgetItem('Anger'))
        self.resTable.setItem(1, 0, QTableWidgetItem('Happiness'))
        self.resTable.setItem(2, 0, QTableWidgetItem('Neutral'))
        self.resTable.setItem(3, 0, QTableWidgetItem('Sadness'))

        self.resTable.resize(220,107)
        self.resTable.move(20,210)
        self.start = False
        self.model_file = 'lin_svc_model_c10.sav'
        self.mins = ''
        self.maxs = ''
        self.means = ''
        self.stds = ''
        self.sample_filename = 'tmp_csv.csv'
        self.wav_filename = 'sample_audio_file.wav'
        self.predicted_class = ''
        self.predicted_prob = ''
        self.model = 'nb_model.sav'
        self.pathDataset = 'Models/BigDataSet/'
        self.loaded_model = joblib.load('Models/BigDataSet/nb_model.sav')
        self.isLongWav = False
        self.duration = []
        self.classes = []
        
        QMetaObject.connectSlotsByName(Form)
        QObject.connect(self.pushButton, SIGNAL("clicked()"), self.startRec)
        QObject.connect(self.stopButton, SIGNAL("clicked()"), self.playRec)

    def proc(self):
        print 'proc'

        self.duration = []
        self.classes = []
        
        files=glob.glob('chunks/*.*')
        for f in files:
            os.remove(f)
        
        p = subprocess.Popen('sox '+ self.wav_filename + ' chunks/out.wav silence 1 0.5 1% 1 0.1 1% : newfile : restart', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout.readlines():
            print line
        retval = p.wait()

        files=sorted(glob.glob('chunks/*.*'))
        labs = []
        for f in files:
            p = subprocess.Popen('soxi -D '+ f, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in p.stdout.readlines():
                if(float(line) == 0.0):
                    print 'remove=', f
                    os.remove(f)
                else:
                    result =  re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", os.path.splitext(ntpath.basename(f))[0])
                    #print 'fdig = ',int(result[0])
                    labs.append(int(result[0]))
                    self.duration.append(float(line))
                    
                print line
            retval = p.wait()

        print 'len dur =',len(self.duration)
        
        files=sorted(glob.glob('chunks/*.*'))
        for f in files:
            print f
            self.wav_filename = f
            self.extract_feature()
            self.predict()

        print 'len classes = ', len(self.classes)

        print 'dur =',(self.duration)
        print 'classes = ', (self.classes)
        print 'sum dur =',sum(self.duration)

        from pylab import *

        figure(1, figsize=(10,10))
        ax = axes([0.1, 0.1, 0.8, 0.8])

        alldur = sum(self.duration)
        colors = []
        for item in self.duration:
            item = (item/alldur) * 100
        print 'per dur = ',self.duration
        print 'len dur = ',sum(self.duration)

        for item in self.classes:
            if item == 1:
               colors.append('red')
            if item == 2:
               colors.append('yellow')
            if item == 3:
               colors.append('green')
            if item == 4:
               colors.append('blue')
        
        labels = labs
        fracs = self.duration 
        pie(fracs,labels = labels,colors=colors, shadow=False, startangle=90,counterclock=False)
        title('Emotion diagram', bbox={'facecolor':'0.8', 'pad':5})
        show()

    def fradio1(self):
        if self.radio1.isChecked():
            self.isLongWav = False 
            self.wav_filename = 'sample_audio_file.wav'
            self.pushButton.setText('Start')

    def fradio2(self):
        if self.radio2.isChecked():
            self.isLongWav = False
            self.pushButton.setText('Process')
            self.wav_filename = os.path.basename(str(self.fileDialog.getOpenFileName()))
    
    def fradio3(self):
        if self.radio3.isChecked():
            self.isLongWav = True
            self.pushButton.setText('Process')
            self.wav_filename = os.path.basename(str(self.fileDialog.getOpenFileName()))

    def slotComboDataSet(self):
        print 'slotComboDataSet(self)'
        if self.dataSetCBox.currentText() == 'Big dataset':
            self.pathDataset = 'Models/BigDataSet/'
            self.loaded_model = joblib.load(self.pathDataset + self.model)
            
            mins  = pd.Series.from_csv('Models/BigDataSet/mins.csv', sep=';', header=None)
            maxs  = pd.Series.from_csv('Models/BigDataSet/maxs.csv', sep=';', header=None)
            means = pd.Series.from_csv('Models/BigDataSet/means.csv',sep=';', header=None)
            stds  = pd.Series.from_csv('Models/BigDataSet/stds.csv', sep=';', header=None)
            
            print self.pathDataset + self.model
        else:
            self.pathDataset = 'Models/SmallDataSet/'
            self.loaded_model = joblib.load(self.pathDataset + self.model)
            
            mins  = pd.Series.from_csv('Models/SmallDataSet/mins.csv', sep=';', header=None)
            maxs  = pd.Series.from_csv('Models/SmallDataSet/maxs.csv', sep=';', header=None)
            means = pd.Series.from_csv('Models/SmallDataSet/means.csv',sep=';', header=None)
            stds  = pd.Series.from_csv('Models/SmallDataSet/stds.csv', sep=';', header=None)
            
            print self.pathDataset + self.model

    def slotComboAlgorithm(self):
        print 'slotComboAlgorithm(self):'
        if self.modelCBox.currentText() == 'Naive Bayes':
            self.model = 'nb_model.sav'
            self.loaded_model = joblib.load(self.pathDataset +'nb_model.sav')
            print self.pathDataset +'nb_model.sav'

        if self.modelCBox.currentText() == 'Linear SVM':
            self.model = 'lin_svc_model.sav'
            self.loaded_model = joblib.load(self.pathDataset +'lin_svc_model.sav')
            print self.pathDataset +'lin_svc_model.sav'
            
        if self.modelCBox.currentText() == 'Polinomial SVM':
            self.model = 'poly_svc_model.sav'
            self.loaded_model = joblib.load(self.pathDataset +'poly_svc_model.sav')
            print self.pathDataset +'poly_svc_model.sav'
            
        if self.modelCBox.currentText() == 'Radial SVM':
            self.model = 'rbf_svc_model.sav'
            self.loaded_model = joblib.load(self.pathDataset +'rbf_svc_model.sav')
            print self.pathDataset +'rbf_svc_model.sav'
        
        if self.modelCBox.currentText() == 'Sigmoid SVM':
            self.model = 'sig_svc_model.sav'
            self.loaded_model = joblib.load(self.pathDataset +'sig_svc_model.sav')
            print self.pathDataset +'sig_svc_model.sav'
        

    def recThread(self,arg):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
        frames = []
        t = threading.currentThread()

        while getattr(t,"do_run",True):
            data = stream.read(CHUNK)
            frames.append(data)
            
        stream.stop_stream()
        stream.close()
        
        waveFile = wave.open(self.wav_filename, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        audio.terminate()
        
    def startRec(self):
        if(self.pushButton.text() == 'Start' or self.pushButton.text() == 'Stop'):
            self.resText.clear()
            if self.start == False:
                self.pushButton.setText("Stop")
                self.thread = threading.Thread(target = self.recThread, args=("task",))
                self.thread.start()
                self.start = True
            else:
                self.pushButton.setText("Start")
                self.thread.do_run = False
                self.thread.join()
                self.start = False
                self.extract_feature()
                self.predict()
        else:
            if self.isLongWav == True:
                self.proc()
            else:
                print 'here'
                self.extract_feature()
                self.predict()

        
    def playRec(self):

        if os.path.exists(self.wav_filename):
            playAudio = PlayAudioFile(self.wav_filename)
            playAudio.play()
            playAudio.close()

    def extract_feature(self):
        p = subprocess.Popen('sudo normalize-audio '+ self.wav_filename, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        retval = p.wait()
        p = subprocess.Popen('/home/aleksandr/diploma/opensmile-2.0-rc1/opensmile/SMILExtract -C /home/aleksandr/diploma/opensmile-2.0-rc1/opensmile/config/emobase2010.conf -I '+ self.wav_filename + ' -O ' + self.sample_filename, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        retval = p.wait()

    def predict(self):
        tmp_d = pd.read_csv(self.sample_filename, sep=' ', skiprows=[0], header=None)
        p = tmp_d[0][1585].split(',')
        
        fl = open("tmp_file.csv","w")
        out = cs.writer(fl, delimiter=';',quoting=cs.QUOTE_ALL)
        out.writerow(p)
        fl.close()
        
        df = pd.read_csv("tmp_file.csv", sep=';', header=None)

        if os.path.exists('tmp_file.csv'):
            os.remove('tmp_file.csv')

        df = df.drop((1583), axis = 1)
        df = df.drop((0), axis = 1)

        df = (df - means)/stds
        df = df.div(np.sqrt(np.square(df).sum(axis=1)), axis=0)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        df = df.astype(float64)
        
        predicted_class = self.loaded_model.predict(df)
        predicted_proba = self.loaded_model.predict_proba(df)

        if self.isLongWav == True:
            self.classes.append(int(predicted_class[0]))
        
        print 'class = ', predicted_class
        print 'class proba = ', predicted_proba
        
        self.resTable.setItem(0, 1, QTableWidgetItem(str(round(predicted_proba[0][0],5))))
        self.resTable.setItem(1, 1, QTableWidgetItem(str(round(predicted_proba[0][1],5))))
        self.resTable.setItem(2, 1, QTableWidgetItem(str(round(predicted_proba[0][2],5))))
        self.resTable.setItem(3, 1, QTableWidgetItem(str(round(predicted_proba[0][3],5))))
        self.resTable.selectRow(np.argmax(predicted_proba[0], axis=0))

        if os.path.exists(self.sample_filename):
            os.remove(self.sample_filename)
        
        del tmp_d
        del p
        del df

if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    Form = QWidget()
    ui = w_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
