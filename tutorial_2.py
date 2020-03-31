# import basic modules
from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
import numpy, random

# get info from the user by a dlg gui
try:  # try to get a previous parameters file
    expInfo = fromFile('lastParams.pickle')
except:  # if not there then use a default set
    expInfo = {'Name':'wqy',
                'Group':0,
                'refOrientation':0}
expInfo['Date'] = data.getDateStr()  # add the current time

# present a dlg to change default params
dlg = gui.DlgFromDict(expInfo, title='SSVEP experiment', fixed=['Date'])
if dlg.OK:
    toFile('lastParams.pickle', expInfo)  # save params to file for next time
else:
    core.quit()  # click cancel to exit

# make a csv file to save info data
fileName = expInfo['Name'] + str(expInfo['Group'])
# simple text file with 'comma-separated-values'
dataFile = open(fileName + '.csv', 'w')
dataFile.write('targetSide, oriIncrement, correct\n')

# create the staircase handler
staircase = data.StairHandler(startVal = 20.0, stepType = 'db',
            # will home in on the 80% threshold
            stepSizes=[8,4,4,2], nUp=1, nDown=3, nTrials=1)

# create window and stimuli
win = visual.Window([1600,900], allowGUI=True, monitor='testMonitor', units='deg')
foil = visual.GratingStim(win, sf=1, size=4, mask='gauss',
                    ori=expInfo['refOrientation'])
target = visual.GratingStim(win, sf=1, size=4, mask='gauss',
                    ori=expInfo['refOrientation'])
fixation = visual.GratingStim(win, color=-1, colorSpace='rgb',
                    tex=None, mask='circle', size=0.2)
# add handy clocks to keep track of time
globalClock = core.Clock()
trialClock = core.Clock()

# display instructions and wait
message1 = visual.TextStim(win, pos=[0,+3], text='Hit a key when ready')
message2 = visual.TextStim(win, pos=[0,-3],
    text='Then press left or right to identify the %.1f deg probe.' %expInfo['refOrientation'])
message1.draw()
message2.draw()
fixation.draw()
win.flip()  # to show stimuli
# pause until there's a new keypress
event.waitKeys()

# control the presentation of stim
for thisIncrement in staircase:  # will continue the staircase until it terminates
    # set location of stim
    targetSide = random.choice([-1,1])  # either +1(right) or -1(left)
    foil.setPos([-5*targetSide, 0])
    target.setPos([5*targetSide, 0])  # in other location
    # set orientation of probe
    foil.setOri(expInfo['refOrientation'] + thisIncrement)
    
    # draw stimuli
    foil.draw()
    target.draw()
    fixation.draw()
    win.flip()
    # wait 500ms; use a loop of x frames for more accurate timing
    core.wait(0.5)
    
    # get input from the subject
    thisResp=None
    while thisResp == None:
        allKeys = event.waitKeys()
        for thisKey in allKeys:
            if thisKey == 'left':
                if targetSide == -1:  # correct response
                    thisResp = 1
                else:  # incorrect response
                    thisResp = -1
            elif thisKey == 'right':
                if targetSide == 1:  # correct response
                    thisResp = 1
                else:  # incorrect response
                    thisResp = -1
            elif thisKey in ['q', 'escape']:  # abort experiment
                core.quit()
        # clear redundant events
        event.clearEvents()
    
    # add the data to the staircase so it can calculate the next level
    staircase.addData(thisResp)
    dataFile.write('%i,%.3f,%i\n' %(targetSide, thisIncrement, thisResp))
    core.wait(1)

# staircase has ended
dataFile.close()
# special python binary file to save all the info
staircase.saveAsPickle(fileName)

# present an output window
print('reversals:')
print(staircase.reversalIntensities)
approxThreshold = numpy.average(staircase.reversalIntensities[-6:])
print('mean of final 6 reversals = %.3f' % (approxThreshold))

# present on-screen feedback
feedback1 = visual.TextStim(win, pos=[0,+3],
        text='mean of final 6 reversals = %.3f' % (approxThreshold))
feedback1.draw()
fixation.draw()
win.flip()
# wait for participant to respond
event.waitKeys()

# end of the experiment
win.close()
core.quit()
