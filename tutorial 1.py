# import basic module from PsychoPy
from psychopy import visual, core, event

# create a window
mywin = visual.Window([1600,900], monitor='testMonitor', units='deg')

# create some stimuli
grating = visual.GratingStim(win=mywin, mask='circle', size=3, pos=[-4,0], sf=3)
fixation = visual.GratingStim(win=mywin, size=0.5, pos=[0,0], sf=0, rgb=-1)

# draw the stimuli and update the window
while True:  # creates a never-ending loop
    # advance phase by 0.05 of a cycle
    grating.setPhase(0.05, '+')
    grating.draw()
    fixation.draw()
    mywin.flip()
    
    if len(event.getKeys())>0:
        break
    event.clearEvents()
    
# clean up
mywin.close()
core.quit()