# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:06:08 2020
An Online High-Frequency SSVEP-BCI System
    This script is based on PsychoPy3 Builder
    
@author: Brynhildr
"""

#%% import 3rd part modules
from psychopy import (gui, visual, core, data, logging, parallel)
from psychopy.constants import (NOT_STARTED, STARTED, FINISHED)
import numpy as np
from numpy import (sin, cos, tan, log, log10, pi, average, sqrt, std, deg2rad, 
                   rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os
import sys
from psychopy.hardware import keyboard

#%%****************************** Before Experiment ******************************%%#
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '3.2.3'
expName = 'SSVEP'  # from the Builder filename that created this script
expInfo = {'Participant': '', 'Session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['Participant'], expInfo['Session'], expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='E:\\Documents\\SSVEP瞬态特性的研究\\Program\\test1.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

#%%****************************** Initialization ******************************%%#
# Start Code: component code to be run before the window creation
# Setup the Window
win = visual.Window(size=[1920, 1080], fullscr=True, screen=0, winType='pyglet',
        allowGUI=False, allowStencil=False, monitor='testMonitor', color=[-1.000,-1.000,-1.000],
        colorSpace='rgb', blendMode='avg', useFBO=True, units='height')
expInfo['frameRate'] = win.getActualFrameRate()  # store frame rate of monitor
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for WelcomePage
WelcomePageClock = core.Clock()
WelcomeText = 'Welcome to participate my experiment'
textWelcome = visual.TextStim(win=win, name='textWelcome', text=WelcomeText,
    font='Arial', pos=(0, 0), height=0.1, wrapWidth=None, ori=0, color='white',
    colorSpace='rgb', opacity=1, languageStyle='LTR',depth=0.0);

# Initialize components for Routine "InstructionPage"
InstructionPageClock = core.Clock()
textInstruction = visual.TextStim(win=win, name='textInstruction',
    text="You will see a series of white square strobes.\n\nEach strobe is about 2 seconds apart.\n\nPress 'space' to continue.\n\n",
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
keyInstruction = keyboard.Keyboard()

# Initialize components for Routine "Blank2000"
Blank2000Clock = core.Clock()
PreparationBlank = visual.TextStim(win=win, name='PreparationBlank',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "ImageShow"
ImageShowClock = core.Clock()
Preparation = visual.TextStim(win=win, name='Preparation',
    text="Here is the square you will see in this experiment\n\nPress 'space' to continue",
    font='Arial',
    pos=(0, 0.3), height=0.05, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
imageShow = visual.ImageStim(
    win=win,
    name='imageShow', units='pix', 
    image='Block.png', mask=None,
    ori=0, pos=(0, 0), size=(250, 250),
    color=[1.000,1.000,1.000], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
keyImageShow = keyboard.Keyboard()

# Initialize components for Routine "Blank2000"
Blank2000Clock = core.Clock()
PreparationBlank = visual.TextStim(win=win, name='PreparationBlank',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "TestTrial"
TestTrialClock = core.Clock()
WhiteBlock = visual.ImageStim(
    win=win,
    name='WhiteBlock', units='pix', 
    image='Block.png', mask=None,
    ori=0, pos=(0, 0), size=(250, 250),
    color='white', colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
TestTrigger = parallel.ParallelPort(address='0x0378')

# Initialize components for Routine "Blank500"
Blank500Clock = core.Clock()
blank500 = visual.TextStim(win=win, name='blank500',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "Blink"
BlinkClock = core.Clock()
blink = visual.TextStim(win=win, name='blink',
    text='Blink',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "Rest1000"
Rest1000Clock = core.Clock()
rest = visual.TextStim(win=win, name='rest',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "Blank2000"
Blank2000Clock = core.Clock()
PreparationBlank = visual.TextStim(win=win, name='PreparationBlank',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "EndScreen"
EndScreenClock = core.Clock()
textEnd = visual.TextStim(win=win, name='textEnd',
    text='Thanks for participating my experiment\nPlease see the researcher now\n',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "WelcomePage"-------
routineTimer.add(2.000000)
# update component parameters for each repeat
# keep track of which components have finished
WelcomePageComponents = [textWelcome]
for thisComponent in WelcomePageComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
WelcomePageClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1
continueRoutine = True

# -------Run Routine "WelcomePage"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = WelcomePageClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=WelcomePageClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *textWelcome* updates
    if textWelcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        textWelcome.frameNStart = frameN  # exact frame index
        textWelcome.tStart = t  # local t and not account for scr refresh
        textWelcome.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(textWelcome, 'tStartRefresh')  # time at next scr refresh
        textWelcome.setAutoDraw(True)
    if textWelcome.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > textWelcome.tStartRefresh + 2.0-frameTolerance:
            # keep track of stop time/frame for later
            textWelcome.tStop = t  # not accounting for scr refresh
            textWelcome.frameNStop = frameN  # exact frame index
            win.timeOnFlip(textWelcome, 'tStopRefresh')  # time at next scr refresh
            textWelcome.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in WelcomePageComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "WelcomePage"-------
for thisComponent in WelcomePageComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('textWelcome.started', textWelcome.tStartRefresh)
thisExp.addData('textWelcome.stopped', textWelcome.tStopRefresh)

# ------Prepare to start Routine "InstructionPage"-------
# update component parameters for each repeat
keyInstruction.keys = []
keyInstruction.rt = []
# keep track of which components have finished
InstructionPageComponents = [textInstruction, keyInstruction]
for thisComponent in InstructionPageComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
InstructionPageClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1
continueRoutine = True

# -------Run Routine "InstructionPage"-------
while continueRoutine:
    # get current time
    t = InstructionPageClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=InstructionPageClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *textInstruction* updates
    if textInstruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        textInstruction.frameNStart = frameN  # exact frame index
        textInstruction.tStart = t  # local t and not account for scr refresh
        textInstruction.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(textInstruction, 'tStartRefresh')  # time at next scr refresh
        textInstruction.setAutoDraw(True)
    
    # *keyInstruction* updates
    waitOnFlip = False
    if keyInstruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        keyInstruction.frameNStart = frameN  # exact frame index
        keyInstruction.tStart = t  # local t and not account for scr refresh
        keyInstruction.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(keyInstruction, 'tStartRefresh')  # time at next scr refresh
        keyInstruction.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(keyInstruction.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(keyInstruction.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if keyInstruction.status == STARTED and not waitOnFlip:
        theseKeys = keyInstruction.getKeys(keyList=['space'], waitRelease=False)
        if len(theseKeys):
            theseKeys = theseKeys[0]  # at least one key was pressed
            
            # check for quit:
            if "escape" == theseKeys:
                endExpNow = True
            keyInstruction.keys = theseKeys.name  # just the last key pressed
            keyInstruction.rt = theseKeys.rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in InstructionPageComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "InstructionPage"-------
for thisComponent in InstructionPageComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('textInstruction.started', textInstruction.tStartRefresh)
thisExp.addData('textInstruction.stopped', textInstruction.tStopRefresh)
# check responses
if keyInstruction.keys in ['', [], None]:  # No response was made
    keyInstruction.keys = None
thisExp.addData('keyInstruction.keys',keyInstruction.keys)
if keyInstruction.keys != None:  # we had a response
    thisExp.addData('keyInstruction.rt', keyInstruction.rt)
thisExp.addData('keyInstruction.started', keyInstruction.tStartRefresh)
thisExp.addData('keyInstruction.stopped', keyInstruction.tStopRefresh)
thisExp.nextEntry()
# the Routine "InstructionPage" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "Blank2000"-------
routineTimer.add(2.000000)
# update component parameters for each repeat
# keep track of which components have finished
Blank2000Components = [PreparationBlank]
for thisComponent in Blank2000Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Blank2000Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1
continueRoutine = True

# -------Run Routine "Blank2000"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = Blank2000Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Blank2000Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *PreparationBlank* updates
    if PreparationBlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        PreparationBlank.frameNStart = frameN  # exact frame index
        PreparationBlank.tStart = t  # local t and not account for scr refresh
        PreparationBlank.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(PreparationBlank, 'tStartRefresh')  # time at next scr refresh
        PreparationBlank.setAutoDraw(True)
    if PreparationBlank.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > PreparationBlank.tStartRefresh + 2.0-frameTolerance:
            # keep track of stop time/frame for later
            PreparationBlank.tStop = t  # not accounting for scr refresh
            PreparationBlank.frameNStop = frameN  # exact frame index
            win.timeOnFlip(PreparationBlank, 'tStopRefresh')  # time at next scr refresh
            PreparationBlank.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Blank2000Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Blank2000"-------
for thisComponent in Blank2000Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('PreparationBlank.started', PreparationBlank.tStartRefresh)
thisExp.addData('PreparationBlank.stopped', PreparationBlank.tStopRefresh)

# ------Prepare to start Routine "ImageShow"-------
# update component parameters for each repeat
keyImageShow.keys = []
keyImageShow.rt = []
# keep track of which components have finished
ImageShowComponents = [Preparation, imageShow, keyImageShow]
for thisComponent in ImageShowComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
ImageShowClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1
continueRoutine = True

# -------Run Routine "ImageShow"-------
while continueRoutine:
    # get current time
    t = ImageShowClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=ImageShowClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Preparation* updates
    if Preparation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        Preparation.frameNStart = frameN  # exact frame index
        Preparation.tStart = t  # local t and not account for scr refresh
        Preparation.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Preparation, 'tStartRefresh')  # time at next scr refresh
        Preparation.setAutoDraw(True)
    
    # *imageShow* updates
    if imageShow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        imageShow.frameNStart = frameN  # exact frame index
        imageShow.tStart = t  # local t and not account for scr refresh
        imageShow.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(imageShow, 'tStartRefresh')  # time at next scr refresh
        imageShow.setAutoDraw(True)
    
    # *keyImageShow* updates
    waitOnFlip = False
    if keyImageShow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        keyImageShow.frameNStart = frameN  # exact frame index
        keyImageShow.tStart = t  # local t and not account for scr refresh
        keyImageShow.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(keyImageShow, 'tStartRefresh')  # time at next scr refresh
        keyImageShow.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(keyImageShow.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(keyImageShow.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if keyImageShow.status == STARTED and not waitOnFlip:
        theseKeys = keyImageShow.getKeys(keyList=['space'], waitRelease=False)
        if len(theseKeys):
            theseKeys = theseKeys[0]  # at least one key was pressed
            
            # check for quit:
            if "escape" == theseKeys:
                endExpNow = True
            keyImageShow.keys = theseKeys.name  # just the last key pressed
            keyImageShow.rt = theseKeys.rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in ImageShowComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "ImageShow"-------
for thisComponent in ImageShowComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('Preparation.started', Preparation.tStartRefresh)
thisExp.addData('Preparation.stopped', Preparation.tStopRefresh)
thisExp.addData('imageShow.started', imageShow.tStartRefresh)
thisExp.addData('imageShow.stopped', imageShow.tStopRefresh)
# check responses
if keyImageShow.keys in ['', [], None]:  # No response was made
    keyImageShow.keys = None
thisExp.addData('keyImageShow.keys',keyImageShow.keys)
if keyImageShow.keys != None:  # we had a response
    thisExp.addData('keyImageShow.rt', keyImageShow.rt)
thisExp.addData('keyImageShow.started', keyImageShow.tStartRefresh)
thisExp.addData('keyImageShow.stopped', keyImageShow.tStopRefresh)
thisExp.nextEntry()
# the Routine "ImageShow" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "Blank2000"-------
routineTimer.add(2.000000)
# update component parameters for each repeat
# keep track of which components have finished
Blank2000Components = [PreparationBlank]
for thisComponent in Blank2000Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Blank2000Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1
continueRoutine = True

# -------Run Routine "Blank2000"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = Blank2000Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Blank2000Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *PreparationBlank* updates
    if PreparationBlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        PreparationBlank.frameNStart = frameN  # exact frame index
        PreparationBlank.tStart = t  # local t and not account for scr refresh
        PreparationBlank.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(PreparationBlank, 'tStartRefresh')  # time at next scr refresh
        PreparationBlank.setAutoDraw(True)
    if PreparationBlank.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > PreparationBlank.tStartRefresh + 2.0-frameTolerance:
            # keep track of stop time/frame for later
            PreparationBlank.tStop = t  # not accounting for scr refresh
            PreparationBlank.frameNStop = frameN  # exact frame index
            win.timeOnFlip(PreparationBlank, 'tStopRefresh')  # time at next scr refresh
            PreparationBlank.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Blank2000Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Blank2000"-------
for thisComponent in Blank2000Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('PreparationBlank.started', PreparationBlank.tStartRefresh)
thisExp.addData('PreparationBlank.stopped', PreparationBlank.tStopRefresh)

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=5, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('TestCondition.xlsx'),
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec('{} = thisTrial[paramName]'.format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "TestTrial"-------
    # update component parameters for each repeat
    # keep track of which components have finished
    TestTrialComponents = [WhiteBlock, TestTrigger]
    for thisComponent in TestTrialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    TestTrialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    continueRoutine = True
    
    # -------Run Routine "TestTrial"-------
    while continueRoutine:
        # get current time
        t = TestTrialClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=TestTrialClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *WhiteBlock* updates
        if WhiteBlock.status == NOT_STARTED and frameN >= 0:
            # keep track of start time/frame for later
            WhiteBlock.frameNStart = frameN  # exact frame index
            WhiteBlock.tStart = t  # local t and not account for scr refresh
            WhiteBlock.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(WhiteBlock, 'tStartRefresh')  # time at next scr refresh
            WhiteBlock.setAutoDraw(True)
        if WhiteBlock.status == STARTED:
            if frameN >= 90:
                # keep track of stop time/frame for later
                WhiteBlock.tStop = t  # not accounting for scr refresh
                WhiteBlock.frameNStop = frameN  # exact frame index
                win.timeOnFlip(WhiteBlock, 'tStopRefresh')  # time at next scr refresh
                WhiteBlock.setAutoDraw(False)
        if WhiteBlock.status == STARTED:  # only update if drawing
            WhiteBlock.setColor([1.000*sin(2*pi*frameN/60*Frequency+Phase*pi),1.000*sin(2*pi*frameN/60*Frequency+Phase*pi),1.000*sin(2*pi*frameN/60*Frequency+Phase*pi)], colorSpace='rgb', log=False)
            WhiteBlock.setOpacity(1, log=False)
        # *TestTrigger* updates
        if TestTrigger.status == NOT_STARTED and frameN >= 0:
            # keep track of start time/frame for later
            TestTrigger.frameNStart = frameN  # exact frame index
            TestTrigger.tStart = t  # local t and not account for scr refresh
            TestTrigger.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(TestTrigger, 'tStartRefresh')  # time at next scr refresh
            TestTrigger.status = STARTED
            win.callOnFlip(TestTrigger.setData, int(ID))
        if TestTrigger.status == STARTED:
            if frameN >= 1:
                # keep track of stop time/frame for later
                TestTrigger.tStop = t  # not accounting for scr refresh
                TestTrigger.frameNStop = frameN  # exact frame index
                win.timeOnFlip(TestTrigger, 'tStopRefresh')  # time at next scr refresh
                TestTrigger.status = FINISHED
                win.callOnFlip(TestTrigger.setData, int(0))
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in TestTrialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "TestTrial"-------
    for thisComponent in TestTrialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('WhiteBlock.started', WhiteBlock.tStartRefresh)
    trials.addData('WhiteBlock.stopped', WhiteBlock.tStopRefresh)
    if TestTrigger.status == STARTED:
        win.callOnFlip(TestTrigger.setData, int(0))
    trials.addData('TestTrigger.started', TestTrigger.tStart)
    trials.addData('TestTrigger.stopped', TestTrigger.tStop)
    # the Routine "TestTrial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "Blank500"-------
    # update component parameters for each repeat
    # keep track of which components have finished
    Blank500Components = [blank500]
    for thisComponent in Blank500Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    Blank500Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    continueRoutine = True
    
    # -------Run Routine "Blank500"-------
    while continueRoutine:
        # get current time
        t = Blank500Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=Blank500Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *blank500* updates
        if blank500.status == NOT_STARTED and frameN >= 0:
            # keep track of start time/frame for later
            blank500.frameNStart = frameN  # exact frame index
            blank500.tStart = t  # local t and not account for scr refresh
            blank500.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(blank500, 'tStartRefresh')  # time at next scr refresh
            blank500.setAutoDraw(True)
        if blank500.status == STARTED:
            if frameN >= 30:
                # keep track of stop time/frame for later
                blank500.tStop = t  # not accounting for scr refresh
                blank500.frameNStop = frameN  # exact frame index
                win.timeOnFlip(blank500, 'tStopRefresh')  # time at next scr refresh
                blank500.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Blank500Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "Blank500"-------
    for thisComponent in Blank500Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('blank500.started', blank500.tStartRefresh)
    trials.addData('blank500.stopped', blank500.tStopRefresh)
    # the Routine "Blank500" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "Blink"-------
    # update component parameters for each repeat
    # keep track of which components have finished
    BlinkComponents = [blink]
    for thisComponent in BlinkComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    BlinkClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    continueRoutine = True
    
    # -------Run Routine "Blink"-------
    while continueRoutine:
        # get current time
        t = BlinkClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=BlinkClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *blink* updates
        if blink.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            blink.frameNStart = frameN  # exact frame index
            blink.tStart = t  # local t and not account for scr refresh
            blink.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(blink, 'tStartRefresh')  # time at next scr refresh
            blink.setAutoDraw(True)
        if blink.status == STARTED:
            if frameN >= 60:
                # keep track of stop time/frame for later
                blink.tStop = t  # not accounting for scr refresh
                blink.frameNStop = frameN  # exact frame index
                win.timeOnFlip(blink, 'tStopRefresh')  # time at next scr refresh
                blink.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in BlinkComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "Blink"-------
    for thisComponent in BlinkComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('blink.started', blink.tStartRefresh)
    trials.addData('blink.stopped', blink.tStopRefresh)
    # the Routine "Blink" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "Rest1000"-------
    # update component parameters for each repeat
    # keep track of which components have finished
    Rest1000Components = [rest]
    for thisComponent in Rest1000Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    Rest1000Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    continueRoutine = True
    
    # -------Run Routine "Rest1000"-------
    while continueRoutine:
        # get current time
        t = Rest1000Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=Rest1000Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rest* updates
        if rest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rest.frameNStart = frameN  # exact frame index
            rest.tStart = t  # local t and not account for scr refresh
            rest.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rest, 'tStartRefresh')  # time at next scr refresh
            rest.setAutoDraw(True)
        if rest.status == STARTED:
            if frameN >= 90:
                # keep track of stop time/frame for later
                rest.tStop = t  # not accounting for scr refresh
                rest.frameNStop = frameN  # exact frame index
                win.timeOnFlip(rest, 'tStopRefresh')  # time at next scr refresh
                rest.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Rest1000Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "Rest1000"-------
    for thisComponent in Rest1000Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('rest.started', rest.tStartRefresh)
    trials.addData('rest.stopped', rest.tStopRefresh)
    # the Routine "Rest1000" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 5 repeats of 'trials'


# ------Prepare to start Routine "Blank2000"-------
routineTimer.add(2.000000)
# update component parameters for each repeat
# keep track of which components have finished
Blank2000Components = [PreparationBlank]
for thisComponent in Blank2000Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
Blank2000Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1
continueRoutine = True

# -------Run Routine "Blank2000"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = Blank2000Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=Blank2000Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *PreparationBlank* updates
    if PreparationBlank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        PreparationBlank.frameNStart = frameN  # exact frame index
        PreparationBlank.tStart = t  # local t and not account for scr refresh
        PreparationBlank.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(PreparationBlank, 'tStartRefresh')  # time at next scr refresh
        PreparationBlank.setAutoDraw(True)
    if PreparationBlank.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > PreparationBlank.tStartRefresh + 2.0-frameTolerance:
            # keep track of stop time/frame for later
            PreparationBlank.tStop = t  # not accounting for scr refresh
            PreparationBlank.frameNStop = frameN  # exact frame index
            win.timeOnFlip(PreparationBlank, 'tStopRefresh')  # time at next scr refresh
            PreparationBlank.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Blank2000Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Blank2000"-------
for thisComponent in Blank2000Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('PreparationBlank.started', PreparationBlank.tStartRefresh)
thisExp.addData('PreparationBlank.stopped', PreparationBlank.tStopRefresh)

# ------Prepare to start Routine "EndScreen"-------
routineTimer.add(15.000000)
# update component parameters for each repeat
# keep track of which components have finished
EndScreenComponents = [textEnd]
for thisComponent in EndScreenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
EndScreenClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1
continueRoutine = True

# -------Run Routine "EndScreen"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = EndScreenClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=EndScreenClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *textEnd* updates
    if textEnd.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        textEnd.frameNStart = frameN  # exact frame index
        textEnd.tStart = t  # local t and not account for scr refresh
        textEnd.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(textEnd, 'tStartRefresh')  # time at next scr refresh
        textEnd.setAutoDraw(True)
    if textEnd.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > textEnd.tStartRefresh + 15-frameTolerance:
            # keep track of stop time/frame for later
            textEnd.tStop = t  # not accounting for scr refresh
            textEnd.frameNStop = frameN  # exact frame index
            win.timeOnFlip(textEnd, 'tStopRefresh')  # time at next scr refresh
            textEnd.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in EndScreenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "EndScreen"-------
for thisComponent in EndScreenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('textEnd.started', textEnd.tStartRefresh)
thisExp.addData('textEnd.stopped', textEnd.tStopRefresh)

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()


