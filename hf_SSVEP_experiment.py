# -*- coding: utf-8 -*-
"""
A SSVEP experiment demo.

Author: Brynhildr
"""

import string
import numpy as np
from ex_base import (Experiment, NeuroScanPort, SSVEP)
from psychopy import (core, visual, event, parallel)


def ssvep(win, port_available=False):
    # when official start, set port_available to True
    # create neuroscan label interface
    if port_available:
        port_address = 0xDEFC
        port = NeuroScanPort(port_address=port_address)

    # config ssvep stimulis
    nElements = 32   # number of the objects
    # the size array of objects(use 2 parameters to describe the size, so create an N*2 array)
    sizes = np.zeros((nElements, 2))
    xys = np.zeros((nElements, 2))    # the position array of objects
    oris = np.zeros((nElements,))    # the orientation array
    sfs = np.ones((nElements,))    # the ... array
    phases = np.zeros((nElements,))    # the phase array
    colors = np.zeros((nElements, 3))    # the color array(use RGB, 3 parameters to describe the color)
    opacities = np.ones((nElements,))    # the opacity array(default 1)
    contrs = np.ones((nElements,))    # the ... array
    winSize = np.array(win.size)    # the window size
    size = np.array([150, 150])    # the size option(pixes)

    # [a,b] means the total number of blocks is a*b
    # so the actual size of the stimulation area is size*shape
    shape = np.array([8, 4])

    # the area for paddings between the stimuli and the screen border
    paddings = (winSize - size*shape)/2

    sizes[:] = size    # set the size of one square

    # config the position for each block, take paddings into consideration
    origin_pos = paddings + size/2
    for i in range(0, shape[1]):
        for j in range(0, shape[0]):
            xys[i*shape[0]+j] = origin_pos+size*[j, i]
    xys = xys - winSize/2
    xys[:, 1] *= -1

    colors[:] = np.array([-1, -1, -1])    # the default background color is gray

    ssvep_stimuli = SSVEP(win)
    ssvep_stimuli.setElements(texture='square', mask=None,
        nElements=nElements, sizes=sizes, xys=xys, oris=oris, sfs=sfs, phases=phases,
        colors=colors, opacities=opacities, contrs=contrs)

    # config text stimulis
    texts = ''.join([string.ascii_uppercase, '_12345'])
    text_stimulis = []
    for text,xy in zip(texts, xys):
        text_stimulis.append(visual.TextStim(win=win, units='pix',
            text=text, pos=xy, height=size[1]/2, wrapWidth=winSize[1],
            color=(0.5, 0.5, 0.5), contrast=1, bold=False, name=text))
    
    # config index stimuli
    index_stimuli = visual.TextStim(win=win, units='pix',
        text='\u25B2', pos=[0, 0], height=size[1]/2, wrapWidth=winSize[1],
        color='yellow', contrast=1, bold=False, name=text)

    # config block stimuli
    block_stimuli = visual.TextStim(win=win, units='pix', text='', pos=[0, 0],
                                    height=size[1]/2, wrapWidth=winSize[1],
                                    color=(0, 0, 0), contrast=1, bold=False)


    for text_stimuli in text_stimulis:
        text_stimuli.draw()
    win.flip()

    # declare specific ssvep parameters
    ssvep_conditions = [{'freq': i, 'phase': j, 'duration': k}
                        for i in np.linspace(60, 15, nElements)
                        for j in [0, 0.5, 1]
                        for k in [0.1, 0.3, 0.5]]
    for i in range(len(ssvep_conditions)):
        ssvep_conditions[i]['id'] = int(i)+1

    ssvep_nrep = 1
    trials = data.TrialHandler(ssvep_conditions, ssvep_nrep,
        name='ssvep', method='random')
    #ssvep_freqs = np.linspace(8, 15, nElements) # 8 -15 HZ
    #ssvep_phases = np.linspace(0, 2, nElements) # different phase

    # seconds for each period
    index_time = 1.5
    lagging_time = 1
    flash_time = 1
    flash_frames = int(flash_time*ssvep_stimuli.refreshRate)
    
    # initialization
    if port_available:
        port.sendLabel(0)

    paradigmClock = core.Clock()
    routineTimer = core.CountdownTimer()
    t = 0
    frameN = 0
    paradigmClock.reset(0)

    # start
    for trial in trials:
        # initialize
        ID = int(trial['id'])
        freq = trial['freq']
        duration = trial['duration']
        phase = trial['phase']
        flash_frames = int(duration * ssvep_stimuli.refreshRate)

        #index_stimuli.setPos(xys[id]+np.array([0, size[1]/2]))
        # phase1:  index period
        routineTimer.reset(0)
        routineTimer.add(index_time)
        while routineTimer.getTime() > 0:
            for text_stimuli in text_stimulis:
                text_stimuli.draw()
            block_stimuli.draw()
            win.flip()
            frameN += 1
        # phase2:  lag before the flash
        routineTimer.reset(0)
        routineTimer.add(lagging_time)
        while routineTimer.getTime() > 0:
            for text_stimuli in text_stimulis:
                text_stimuli.draw()
            win.flip()
            frameN += 1
        # phase3: flashing
        for i in range(flash_frames):
            if port_available:
                #win.callOnFlip(port.sendLabel, id + 1)
                win.callOnFlip(port.sendLabel, ID)
            ssvep_stimuli.update(i, freq, phase)
            for text_stimuli in text_stimulis:
                text_stimuli.draw()
            win.flip()
            frameN += 1
        # phase4:  lag before the next flash
        routineTimer.reset(0)

        routineTimer.add(lagging_time)
        while routineTimer.getTime() > 0:
            pass
        
        if event.getKeys('backspace'):
            print('Force stopped by user.')
            break

    t = paradigmClock.getTime()
    return frameN, t

if __name__ == '__main__':
    # clarify monitor information
    mon = monitors.Monitor(name='swolf_mon', width=53.704, distance=45, gamma=None, verbose=False, autoLog=False)
    mon.setSizePix([1920, 1080])
    mon.save()

    # register self-defined paradigm and run experiment
    ex = Experiment()
    ex.setMonitor(mon)
    ex.registerParadigm("SSVEP", ssvep, port_available=False)
    ex.run()