# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:20:17 2020

PsychoPy test program

@author: Brynhildr
"""

# import basic model
from psychopy import visual, core

# set window object
win = visual.Window([1600,900])

# double-shuffle: render TextStim interfacce (window object, text='')
message = visual.TextStim(win, text='hello')
# automatically draw every frame
message.autoDraw = True
# set position of text stim
message.pos = [-0.5, 0.2]
# rotate stim 90 degrees
message.ori = 90
# make the stim slightly transparent (take effect in next one)
message.opacity = 0.2
# resize stim (not meaningful for strings)
message.size = 5
# refresh the screen to show the pre-rendered interface
win.flip()
# keep this interface for 2s
core.wait(2.0)

# render next interface (double-shuffle)
# change properties of existing stim
# NOT RESET！ CHANGE EXISTING ONE
message.text = 'world'
message.pos = [0.5, -0.2]
message.ori = 0
message.size += 2  #(not meaningful for strings)
win.flip()
core.wait(2.0)