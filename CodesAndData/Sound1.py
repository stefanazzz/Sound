#!/usr/bin/env python
# coding: utf-8
'''
On my computer I have a "sound" environment and spyder5.
"sound" has obspy, librosa, simpleaudio & spyder5 plus other standard modules. 
So I initialise the system with: 
 conda activate sound
And then launch spyder5 simply as:
 spyder5
'''
import obspy
import obspy.signal
from obspy import read
from obspy.imaging.spectrogram import spectrogram
from obspy.signal.cross_correlation import correlate,xcorr_max
from IPython.display import display, Markdown
import numpy as np
import copy
#get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
import librosa 
import warnings
warnings.filterwarnings('ignore')
import os as os
from scipy.io.wavfile import write as swrite
from IPython.display import Audio
import subprocess
#%% START BY LOADING A FILE AND PLOTTING
os.chdir('/data/Sound')

''' read data file with obspy read, then print file info:'''
st=read("2011-03-11-mw91-near-east-coast-of-honshu-japan-2.miniseed")
print(st)
''' file contains 6 traces; show second one of 6: st[1]'''

''' plot trace and spectrogram using obspy.
    - To show graphics in plots pane select:
    Preferences > IPython Console > Graphics > Backend > Inline 
    Cycle that once to take effect for some strange reason.
    - To show plots in floating window:
    Preferences > IPython Console > Graphics > Backend > Qt5
        or:
    Preferences > IPython Console > Graphics > Backend > Automatic'''
#plt.close('all')
fig = st[0].plot(show=False)
fig.show()  # Shows the figure in a new window
#%%
fig = st[1].spectrogram(log=True, wlen=500, clip = [.01, .2], show=False)
fig.show()
#%% EXTRACT INFO AND DATA:
dt=st[1].stats.delta; # extract time step, comopute sampling rate
sr=int(1/dt)
window = signal.windows.tukey(len(data),alpha = 0.02) # taper trace ends 
data=st[1].data # extract data in array format
data = data * window
#%% START THE PROCESS TO PRDUCE A SOUND:
''' normalise then set to 16 bits '''
scaled = np.int16(data/np.max(np.abs(data)) * 32767)
''' 1) CONVERT TO SOUND WITH NO PROCESSING 
    write 'as is' to file wav 
    second arg is assumend sampling rate
    sampling rate scales the duration of sound (larger sr = faster)''' 
swrite('test1.wav', 44100, scaled)
#write('test1.wav', 80000, scaled)
#%% CALL THE OPERATING SYSTEM TO USE YOUR FAVOURITE AUDIO PLAYER:
'''NOTE: aplay is a linux player. You will use another for windows or mac, eg:
     subprocess.Popen(["start", "test1.wav"], shell=True) # for Windows
     subprocess.Popen(["afplay", "test1.wav"], shell=True) # for mac
# or install ffmpeg (free) on ANY SYSTEM and use:
     subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "audiofile.wav"])'''
subprocess.Popen(["aplay", "test1.wav"])
#%%
''' 2) CONVERT TO SOUND WITH PITCH SHIFT '''
y, sr = librosa.load('test1.wav', sr=44100)
''' shift pitch -- 12 steps is one octave '''
yshift=librosa.effects.pitch_shift(y, sr= sr,  n_steps= +24)
#yshift=np.multiply(yshift,1) alter amplitude
swrite('honshu_pitched.wav',44100,yshift[0:int(len(yshift)/1)])
subprocess.Popen(["aplay", 'honshu_pitched.wav'])
#%%
# Another file
st=read("2011-03-11-mw91-near-east-coast-of-honshu-japan-3_long.miniseed")

data=st[1].data
dt=st[1].stats.delta; sr=int(1/dt)
window = signal.windows.tukey(len(data),alpha = 0.02)
data = data * window
scaled = np.int16(data/np.max(np.abs(data)) * 32767)
swrite('test1b.wav', 44100, scaled)
subprocess.Popen(['aplay','test1b.wav'])
#%%
y, sr = librosa.load('test1b.wav', sr=44100)
#yshift=librosa.effects.pitch_shift(y, sr, n_steps=+15)
yshift=librosa.effects.pitch_shift(y, sr= sr, n_steps=+24)
#swrite('TohokuSurfStretched.wav',int(.5e5),yshift[0:int(len(yshift)/1)])
yshift=np.multiply(yshift,2) # increase amplitude by factor of .
swrite('TohokuSurfStretched2.wav',44100,yshift[0:len(yshift)])
subprocess.Popen(['aplay','TohokuSurfStretched2.wav'])

#%%
y, sr = librosa.load('test1.wav', sr=44100)
yshift=librosa.effects.pitch_shift(y, sr=sr, n_steps=+15)
proc = subprocess.Popen(
    ["aplay", "-f", "S16_LE", "-r", str(sr), "-c", "1"],
    stdin=subprocess.PIPE)
proc.stdin.write(yshift.tobytes())
proc.stdin.close()
#%%
plt.close('all')
fig = st[1].spectrogram(log=True, wlen=500, dbscale= True, show=False)#,  clip = [.01, .05]) # get audio write and play libraries :'''
fig.show()
fig =st[1].spectrogram(log=True, wlen=500, dbscale= False, clip = [.01, .05],show=False) #,  clip = [.01, .05]) # get audio write and play libraries :'''
fig.show()
#%%

nis=read("HL.NISR..HHZ.D.2014.247.235957.SAC")
nata=nis[0].data
window = signal.windows.tukey(len(nata),alpha = 0.02)
nata = nata * window
scaled=np.int16(nata/np.max(np.abs(nata))* 32767)
swrite('test2.wav', 44100, scaled)
y, sr = librosa.load('test2.wav', sr=44100)
yshift=librosa.effects.pitch_shift(y, sr = sr, n_steps=-60)
scaled=np.int16(yshift/np.max(np.abs(yshift))* 32767)
swrite('test3.wav', 44100, scaled)

#%%
nata=nis[0].data
scaled=np.int16(nata/np.max(np.abs(nata))* 32767)
swrite('test2b.wav', 22050, scaled)

#%%
nis=read("HL.NISR..HHZ.D.2014.247.235957.SAC")
nata=nis[0].data
scaled=np.int16(nata/np.max(np.abs(nata))* 32767)
swrite('test2.wav', 44100, scaled)
y, sr = librosa.load('test2.wav', sr=44100)
yshift=librosa.effects.pitch_shift(y, sr=sr, n_steps=-30)
scaled=np.int16(yshift/np.max(np.abs(yshift))* 32767)
swrite('test4.wav', 44100, scaled)

#%%

test_nn=read("TEST4.mseed")

nata=test_nn[0].data
scaled=np.int16(nata/np.max(np.abs(nata))* 32767)
swrite('test_nn4.wav', 44100, scaled)
y, sr = librosa.load('test5.wav', sr=44100)
yshift=librosa.effects.pitch_shift(y, sr=sr, n_steps=-30)
scaled=np.int16(yshift/np.max(np.abs(yshift))* 32767)                 
swrite('test_nn4b.wav', 44100, scaled)
proc = subprocess.Popen(
    ["aplay", "-f", "S16_LE", "-r", str(sr), "-c", "1"],
    stdin=subprocess.PIPE)
proc.stdin.write(scaled.tobytes())
proc.stdin.close()

