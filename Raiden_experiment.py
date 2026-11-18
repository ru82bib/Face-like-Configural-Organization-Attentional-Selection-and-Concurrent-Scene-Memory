#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.2.4),
    on November 18, 2026, at 23:43
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.2.4'
expName = 'Raiden_experiment'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='D:\\Master\\Psychophysics\\code\\Face-like-Configural-Organization-Attentional-Selection-and-Concurrent-Scene-Memory\\Raiden_experiment.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Welcome" ---
    text = visual.TextStim(win=win, name='text',
        text="Welcome to Raiden's Experiment! \n\nPlease follow the instructions on the next page. \n\nPress 'SPACE' to move to the next page.",
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    
    # --- Initialize components for Routine "Instructions_for_practice" ---
    instr_text = visual.TextStim(win=win, name='instr_text',
        text="In this experiment, you will press 'SPACE' when you see the target.\n\nThe target for this block is 'face-like' images.\n\nExamples of a target and a non-target are shown below.\n\nTo start the practice, please press 'SPACE'.",
        font='Arial',
        pos=[0, 0.3], draggable=False, height=0.035, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    ex_target = visual.ImageStim(
        win=win,
        name='ex_target', 
        image='practice/target_1.png', mask=None, anchor='center',
        ori=0.0, pos=[-0.3, -0.22], draggable=False, size=[0.3, 0.3],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    intr_key = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    label_target = visual.TextStim(win=win, name='label_target',
        text='TARGET',
        font='Arial',
        pos=[-0.3, -0.02], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    label_nontarget = visual.TextStim(win=win, name='label_nontarget',
        text='NON-TARGET',
        font='Arial',
        pos=[0.3, -0.02], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    ex_nontarget = visual.ImageStim(
        win=win,
        name='ex_nontarget', 
        image='practice/non_target_1.png', mask=None, anchor='center',
        ori=0.0, pos=[0.3, -0.22], draggable=False, size=[0.3, 0.3],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    
    # --- Initialize components for Routine "practice_fixation1" ---
    fixation1 = visual.TextStim(win=win, name='fixation1',
        text='+',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Practice_Trial" ---
    Background = visual.ImageStim(
        win=win,
        name='Background', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0, 0], draggable=False, size=[0.8, 0.8],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    Central = visual.ImageStim(
        win=win,
        name='Central', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0, 0], draggable=False, size=[0.15, 0.15],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    response = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    
    # --- Initialize components for Routine "End_practice" ---
    end_practice = visual.TextStim(win=win, name='end_practice',
        text='Practice ended\n\nplease click right with mouse',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "practice_fixation" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text="Please press the number of the image that you saw in the first session of the experiment\n\nPress 'SPACE' to start next trial",
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    fix = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    
    # --- Initialize components for Routine "fixation_cross" ---
    fix_cross = visual.TextStim(win=win, name='fix_cross',
        text='+',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "recognition_trial" ---
    opt1 = visual.ImageStim(
        win=win,
        name='opt1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[-0.6, 0], draggable=False, size=[0.36, 0.36],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    opt2 = visual.ImageStim(
        win=win,
        name='opt2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[-0.2, 0], draggable=False, size=[0.36, 0.36],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    opt3 = visual.ImageStim(
        win=win,
        name='opt3', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0.2, 0], draggable=False, size=[0.36, 0.36],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    opt4 = visual.ImageStim(
        win=win,
        name='opt4', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0.6, 0], draggable=False, size=[0.36, 0.36],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    rec_key = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    var_1 = visual.TextStim(win=win, name='var_1',
        text='1',
        font='Arial',
        pos=[-0.6, -0.25], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    var_2 = visual.TextStim(win=win, name='var_2',
        text='2',
        font='Arial',
        pos=[-0.2, -0.25], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    var_3 = visual.TextStim(win=win, name='var_3',
        text='3',
        font='Arial',
        pos=[0.2, -0.25], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    var_4 = visual.TextStim(win=win, name='var_4',
        text='4',
        font='Arial',
        pos=[0.6, -0.25], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    
    # --- Initialize components for Routine "End_practice1" ---
    end_practice1 = visual.TextStim(win=win, name='end_practice1',
        text='Practice ended\n\nplease click right with mouse',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mouse1 = event.Mouse(win=win)
    x, y = [None, None]
    mouse1.mouseClock = core.Clock()
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Welcome" ---
    # create an object to store info about Routine Welcome
    Welcome = data.Routine(
        name='Welcome',
        components=[text, key_resp],
    )
    Welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for Welcome
    Welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Welcome.tStart = globalClock.getTime(format='float')
    Welcome.status = STARTED
    thisExp.addData('Welcome.started', Welcome.tStart)
    Welcome.maxDuration = None
    # keep track of which components have finished
    WelcomeComponents = Welcome.components
    for thisComponent in Welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome" ---
    thisExp.currentRoutine = Welcome
    Welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Welcome,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Welcome.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Welcome.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome" ---
    for thisComponent in Welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Welcome
    Welcome.tStop = globalClock.getTime(format='float')
    Welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Welcome.stopped', Welcome.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "Welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Instructions_for_practice" ---
    # create an object to store info about Routine Instructions_for_practice
    Instructions_for_practice = data.Routine(
        name='Instructions_for_practice',
        components=[instr_text, ex_target, intr_key, label_target, label_nontarget, ex_nontarget],
    )
    Instructions_for_practice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for intr_key
    intr_key.keys = []
    intr_key.rt = []
    _intr_key_allKeys = []
    # store start times for Instructions_for_practice
    Instructions_for_practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instructions_for_practice.tStart = globalClock.getTime(format='float')
    Instructions_for_practice.status = STARTED
    thisExp.addData('Instructions_for_practice.started', Instructions_for_practice.tStart)
    Instructions_for_practice.maxDuration = None
    # keep track of which components have finished
    Instructions_for_practiceComponents = Instructions_for_practice.components
    for thisComponent in Instructions_for_practice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instructions_for_practice" ---
    thisExp.currentRoutine = Instructions_for_practice
    Instructions_for_practice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr_text* updates
        
        # if instr_text is starting this frame...
        if instr_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            instr_text.frameNStart = frameN  # exact frame index
            instr_text.tStart = t  # local t and not account for scr refresh
            instr_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_text.started')
            # update status
            instr_text.status = STARTED
            instr_text.setAutoDraw(True)
        
        # if instr_text is active this frame...
        if instr_text.status == STARTED:
            # update params
            pass
        
        # *ex_target* updates
        
        # if ex_target is starting this frame...
        if ex_target.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            ex_target.frameNStart = frameN  # exact frame index
            ex_target.tStart = t  # local t and not account for scr refresh
            ex_target.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ex_target, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'ex_target.started')
            # update status
            ex_target.status = STARTED
            ex_target.setAutoDraw(True)
        
        # if ex_target is active this frame...
        if ex_target.status == STARTED:
            # update params
            pass
        
        # *intr_key* updates
        waitOnFlip = False
        
        # if intr_key is starting this frame...
        if intr_key.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            intr_key.frameNStart = frameN  # exact frame index
            intr_key.tStart = t  # local t and not account for scr refresh
            intr_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intr_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intr_key.started')
            # update status
            intr_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intr_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intr_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intr_key.status == STARTED and not waitOnFlip:
            theseKeys = intr_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intr_key_allKeys.extend(theseKeys)
            if len(_intr_key_allKeys):
                intr_key.keys = _intr_key_allKeys[-1].name  # just the last key pressed
                intr_key.rt = _intr_key_allKeys[-1].rt
                intr_key.duration = _intr_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *label_target* updates
        
        # if label_target is starting this frame...
        if label_target.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            label_target.frameNStart = frameN  # exact frame index
            label_target.tStart = t  # local t and not account for scr refresh
            label_target.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label_target, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label_target.started')
            # update status
            label_target.status = STARTED
            label_target.setAutoDraw(True)
        
        # if label_target is active this frame...
        if label_target.status == STARTED:
            # update params
            pass
        
        # *label_nontarget* updates
        
        # if label_nontarget is starting this frame...
        if label_nontarget.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            label_nontarget.frameNStart = frameN  # exact frame index
            label_nontarget.tStart = t  # local t and not account for scr refresh
            label_nontarget.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(label_nontarget, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'label_nontarget.started')
            # update status
            label_nontarget.status = STARTED
            label_nontarget.setAutoDraw(True)
        
        # if label_nontarget is active this frame...
        if label_nontarget.status == STARTED:
            # update params
            pass
        
        # *ex_nontarget* updates
        
        # if ex_nontarget is starting this frame...
        if ex_nontarget.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            ex_nontarget.frameNStart = frameN  # exact frame index
            ex_nontarget.tStart = t  # local t and not account for scr refresh
            ex_nontarget.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ex_nontarget, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'ex_nontarget.started')
            # update status
            ex_nontarget.status = STARTED
            ex_nontarget.setAutoDraw(True)
        
        # if ex_nontarget is active this frame...
        if ex_nontarget.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Instructions_for_practice,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Instructions_for_practice.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Instructions_for_practice.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Instructions_for_practice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instructions_for_practice" ---
    for thisComponent in Instructions_for_practice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instructions_for_practice
    Instructions_for_practice.tStop = globalClock.getTime(format='float')
    Instructions_for_practice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instructions_for_practice.stopped', Instructions_for_practice.tStop)
    # check responses
    if intr_key.keys in ['', [], None]:  # No response was made
        intr_key.keys = None
    thisExp.addData('intr_key.keys',intr_key.keys)
    if intr_key.keys != None:  # we had a response
        thisExp.addData('intr_key.rt', intr_key.rt)
        thisExp.addData('intr_key.duration', intr_key.duration)
    thisExp.nextEntry()
    # the Routine "Instructions_for_practice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice = data.TrialHandler2(
        name='practice',
        nReps=1, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('practice_trial.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(practice)  # add the loop to the experiment
    thisPractice = practice.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
    if thisPractice != None:
        for paramName in thisPractice:
            globals()[paramName] = thisPractice[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice in practice:
        practice.status = STARTED
        if hasattr(thisPractice, 'status'):
            thisPractice.status = STARTED
        currentLoop = practice
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
        if thisPractice != None:
            for paramName in thisPractice:
                globals()[paramName] = thisPractice[paramName]
        
        # --- Prepare to start Routine "practice_fixation1" ---
        # create an object to store info about Routine practice_fixation1
        practice_fixation1 = data.Routine(
            name='practice_fixation1',
            components=[fixation1],
        )
        practice_fixation1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for practice_fixation1
        practice_fixation1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_fixation1.tStart = globalClock.getTime(format='float')
        practice_fixation1.status = STARTED
        thisExp.addData('practice_fixation1.started', practice_fixation1.tStart)
        practice_fixation1.maxDuration = None
        # keep track of which components have finished
        practice_fixation1Components = practice_fixation1.components
        for thisComponent in practice_fixation1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_fixation1" ---
        thisExp.currentRoutine = practice_fixation1
        practice_fixation1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisPractice, 'status') and thisPractice.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation1* updates
            
            # if fixation1 is starting this frame...
            if fixation1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                fixation1.frameNStart = frameN  # exact frame index
                fixation1.tStart = t  # local t and not account for scr refresh
                fixation1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation1.started')
                # update status
                fixation1.status = STARTED
                fixation1.setAutoDraw(True)
            
            # if fixation1 is active this frame...
            if fixation1.status == STARTED:
                # update params
                pass
            
            # if fixation1 is stopping this frame...
            if fixation1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation1.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation1.tStop = t  # not accounting for scr refresh
                    fixation1.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation1.stopped')
                    # update status
                    fixation1.status = FINISHED
                    fixation1.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=practice_fixation1,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                practice_fixation1.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if practice_fixation1.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in practice_fixation1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_fixation1" ---
        for thisComponent in practice_fixation1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_fixation1
        practice_fixation1.tStop = globalClock.getTime(format='float')
        practice_fixation1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_fixation1.stopped', practice_fixation1.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if practice_fixation1.maxDurationReached:
            routineTimer.addTime(-practice_fixation1.maxDuration)
        elif practice_fixation1.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "Practice_Trial" ---
        # create an object to store info about Routine Practice_Trial
        Practice_Trial = data.Routine(
            name='Practice_Trial',
            components=[Background, Central, response],
        )
        Practice_Trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Background.setImage(practice_img)
        Central.setImage(central_img)
        # create starting attributes for response
        response.keys = []
        response.rt = []
        _response_allKeys = []
        # store start times for Practice_Trial
        Practice_Trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Practice_Trial.tStart = globalClock.getTime(format='float')
        Practice_Trial.status = STARTED
        thisExp.addData('Practice_Trial.started', Practice_Trial.tStart)
        Practice_Trial.maxDuration = None
        # keep track of which components have finished
        Practice_TrialComponents = Practice_Trial.components
        for thisComponent in Practice_Trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Practice_Trial" ---
        thisExp.currentRoutine = Practice_Trial
        Practice_Trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPractice, 'status') and thisPractice.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Background* updates
            
            # if Background is starting this frame...
            if Background.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Background.frameNStart = frameN  # exact frame index
                Background.tStart = t  # local t and not account for scr refresh
                Background.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Background, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Background.started')
                # update status
                Background.status = STARTED
                Background.setAutoDraw(True)
            
            # if Background is active this frame...
            if Background.status == STARTED:
                # update params
                pass
            
            # if Background is stopping this frame...
            if Background.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Background.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Background.tStop = t  # not accounting for scr refresh
                    Background.tStopRefresh = tThisFlipGlobal  # on global time
                    Background.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Background.stopped')
                    # update status
                    Background.status = FINISHED
                    Background.setAutoDraw(False)
            
            # *Central* updates
            
            # if Central is starting this frame...
            if Central.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Central.frameNStart = frameN  # exact frame index
                Central.tStart = t  # local t and not account for scr refresh
                Central.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Central, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Central.started')
                # update status
                Central.status = STARTED
                Central.setAutoDraw(True)
            
            # if Central is active this frame...
            if Central.status == STARTED:
                # update params
                pass
            
            # if Central is stopping this frame...
            if Central.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Central.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Central.tStop = t  # not accounting for scr refresh
                    Central.tStopRefresh = tThisFlipGlobal  # on global time
                    Central.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Central.stopped')
                    # update status
                    Central.status = FINISHED
                    Central.setAutoDraw(False)
            
            # *response* updates
            waitOnFlip = False
            
            # if response is starting this frame...
            if response.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                response.frameNStart = frameN  # exact frame index
                response.tStart = t  # local t and not account for scr refresh
                response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'response.started')
                # update status
                response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(response.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(response.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if response.status == STARTED and not waitOnFlip:
                theseKeys = response.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _response_allKeys.extend(theseKeys)
                if len(_response_allKeys):
                    response.keys = _response_allKeys[-1].name  # just the last key pressed
                    response.rt = _response_allKeys[-1].rt
                    response.duration = _response_allKeys[-1].duration
                    # was this correct?
                    if (response.keys == str('correct_key')) or (response.keys == 'correct_key'):
                        response.corr = 1
                    else:
                        response.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from code
            if t >= 0.5:
                continueRoutine = False
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Practice_Trial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Practice_Trial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Practice_Trial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Practice_Trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Practice_Trial" ---
        for thisComponent in Practice_Trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Practice_Trial
        Practice_Trial.tStop = globalClock.getTime(format='float')
        Practice_Trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Practice_Trial.stopped', Practice_Trial.tStop)
        # check responses
        if response.keys in ['', [], None]:  # No response was made
            response.keys = None
            # was no response the correct answer?!
            if str('correct_key').lower() == 'none':
               response.corr = 1;  # correct non-response
            else:
               response.corr = 0;  # failed to respond (incorrectly)
        # store data for practice (TrialHandler)
        practice.addData('response.keys',response.keys)
        practice.addData('response.corr', response.corr)
        if response.keys != None:  # we had a response
            practice.addData('response.rt', response.rt)
            practice.addData('response.duration', response.duration)
        # the Routine "Practice_Trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisPractice as finished
        if hasattr(thisPractice, 'status'):
            thisPractice.status = FINISHED
        # if awaiting a pause, pause now
        if practice.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            practice.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'practice'
    practice.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "End_practice" ---
    # create an object to store info about Routine End_practice
    End_practice = data.Routine(
        name='End_practice',
        components=[end_practice, mouse],
    )
    End_practice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse
    mouse.x = []
    mouse.y = []
    mouse.leftButton = []
    mouse.midButton = []
    mouse.rightButton = []
    mouse.time = []
    gotValidClick = False  # until a click is received
    # store start times for End_practice
    End_practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    End_practice.tStart = globalClock.getTime(format='float')
    End_practice.status = STARTED
    thisExp.addData('End_practice.started', End_practice.tStart)
    End_practice.maxDuration = None
    # keep track of which components have finished
    End_practiceComponents = End_practice.components
    for thisComponent in End_practice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "End_practice" ---
    thisExp.currentRoutine = End_practice
    End_practice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_practice* updates
        
        # if end_practice is starting this frame...
        if end_practice.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            end_practice.frameNStart = frameN  # exact frame index
            end_practice.tStart = t  # local t and not account for scr refresh
            end_practice.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_practice, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_practice.started')
            # update status
            end_practice.status = STARTED
            end_practice.setAutoDraw(True)
        
        # if end_practice is active this frame...
        if end_practice.status == STARTED:
            # update params
            pass
        # *mouse* updates
        
        # if mouse is starting this frame...
        if mouse.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            mouse.frameNStart = frameN  # exact frame index
            mouse.tStart = t  # local t and not account for scr refresh
            mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse.started', t)
            # update status
            mouse.status = STARTED
            mouse.mouseClock.reset()
            # if button is down already this ISN'T a new click
            mouse.prevButtonState = mouse.getPressed()
        if mouse.status == STARTED:  # only update if started and not finished!
            buttons = mouse.getPressed()
            # button state changed?
            if buttons != mouse.prevButtonState:
                mouse.prevButtonState = buttons
                # state changed to a new click
                if sum(buttons) > 0:
                    pass
                    x, y = mouse.getPos()
                    mouse.x.append(float(x))
                    mouse.y.append(float(y))
                    buttons = mouse.getPressed()
                    mouse.leftButton.append(buttons[0])
                    mouse.midButton.append(buttons[1])
                    mouse.rightButton.append(buttons[2])
                    mouse.time.append(mouse.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=End_practice,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            End_practice.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if End_practice.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in End_practice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End_practice" ---
    for thisComponent in End_practice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for End_practice
    End_practice.tStop = globalClock.getTime(format='float')
    End_practice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('End_practice.stopped', End_practice.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse.x', mouse.x)
    thisExp.addData('mouse.y', mouse.y)
    thisExp.addData('mouse.leftButton', mouse.leftButton)
    thisExp.addData('mouse.midButton', mouse.midButton)
    thisExp.addData('mouse.rightButton', mouse.rightButton)
    thisExp.addData('mouse.time', mouse.time)
    thisExp.nextEntry()
    # the Routine "End_practice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "practice_fixation" ---
    # create an object to store info about Routine practice_fixation
    practice_fixation = data.Routine(
        name='practice_fixation',
        components=[fixation, fix],
    )
    practice_fixation.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for fix
    fix.keys = []
    fix.rt = []
    _fix_allKeys = []
    # store start times for practice_fixation
    practice_fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    practice_fixation.tStart = globalClock.getTime(format='float')
    practice_fixation.status = STARTED
    thisExp.addData('practice_fixation.started', practice_fixation.tStart)
    practice_fixation.maxDuration = None
    # keep track of which components have finished
    practice_fixationComponents = practice_fixation.components
    for thisComponent in practice_fixation.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "practice_fixation" ---
    thisExp.currentRoutine = practice_fixation
    practice_fixation.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation* updates
        
        # if fixation is starting this frame...
        if fixation.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            fixation.frameNStart = frameN  # exact frame index
            fixation.tStart = t  # local t and not account for scr refresh
            fixation.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fixation.started')
            # update status
            fixation.status = STARTED
            fixation.setAutoDraw(True)
        
        # if fixation is active this frame...
        if fixation.status == STARTED:
            # update params
            pass
        
        # *fix* updates
        waitOnFlip = False
        
        # if fix is starting this frame...
        if fix.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            fix.frameNStart = frameN  # exact frame index
            fix.tStart = t  # local t and not account for scr refresh
            fix.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fix, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fix.started')
            # update status
            fix.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(fix.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(fix.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if fix.status == STARTED and not waitOnFlip:
            theseKeys = fix.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _fix_allKeys.extend(theseKeys)
            if len(_fix_allKeys):
                fix.keys = _fix_allKeys[-1].name  # just the last key pressed
                fix.rt = _fix_allKeys[-1].rt
                fix.duration = _fix_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=practice_fixation,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            practice_fixation.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if practice_fixation.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in practice_fixation.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice_fixation" ---
    for thisComponent in practice_fixation.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for practice_fixation
    practice_fixation.tStop = globalClock.getTime(format='float')
    practice_fixation.tStopRefresh = tThisFlipGlobal
    thisExp.addData('practice_fixation.stopped', practice_fixation.tStop)
    # check responses
    if fix.keys in ['', [], None]:  # No response was made
        fix.keys = None
    thisExp.addData('fix.keys',fix.keys)
    if fix.keys != None:  # we had a response
        thisExp.addData('fix.rt', fix.rt)
        thisExp.addData('fix.duration', fix.duration)
    thisExp.nextEntry()
    # the Routine "practice_fixation" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    recognition_loop = data.TrialHandler2(
        name='recognition_loop',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('recognition.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(recognition_loop)  # add the loop to the experiment
    thisRecognition_loop = recognition_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRecognition_loop.rgb)
    if thisRecognition_loop != None:
        for paramName in thisRecognition_loop:
            globals()[paramName] = thisRecognition_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRecognition_loop in recognition_loop:
        recognition_loop.status = STARTED
        if hasattr(thisRecognition_loop, 'status'):
            thisRecognition_loop.status = STARTED
        currentLoop = recognition_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRecognition_loop.rgb)
        if thisRecognition_loop != None:
            for paramName in thisRecognition_loop:
                globals()[paramName] = thisRecognition_loop[paramName]
        
        # --- Prepare to start Routine "fixation_cross" ---
        # create an object to store info about Routine fixation_cross
        fixation_cross = data.Routine(
            name='fixation_cross',
            components=[fix_cross],
        )
        fixation_cross.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fixation_cross
        fixation_cross.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation_cross.tStart = globalClock.getTime(format='float')
        fixation_cross.status = STARTED
        thisExp.addData('fixation_cross.started', fixation_cross.tStart)
        fixation_cross.maxDuration = None
        # keep track of which components have finished
        fixation_crossComponents = fixation_cross.components
        for thisComponent in fixation_cross.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_cross" ---
        thisExp.currentRoutine = fixation_cross
        fixation_cross.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisRecognition_loop, 'status') and thisRecognition_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix_cross* updates
            
            # if fix_cross is starting this frame...
            if fix_cross.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                fix_cross.frameNStart = frameN  # exact frame index
                fix_cross.tStart = t  # local t and not account for scr refresh
                fix_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_cross.started')
                # update status
                fix_cross.status = STARTED
                fix_cross.setAutoDraw(True)
            
            # if fix_cross is active this frame...
            if fix_cross.status == STARTED:
                # update params
                pass
            
            # if fix_cross is stopping this frame...
            if fix_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_cross.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_cross.tStop = t  # not accounting for scr refresh
                    fix_cross.tStopRefresh = tThisFlipGlobal  # on global time
                    fix_cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_cross.stopped')
                    # update status
                    fix_cross.status = FINISHED
                    fix_cross.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=fixation_cross,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixation_cross.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixation_cross.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixation_cross.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_cross" ---
        for thisComponent in fixation_cross.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation_cross
        fixation_cross.tStop = globalClock.getTime(format='float')
        fixation_cross.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation_cross.stopped', fixation_cross.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_cross.maxDurationReached:
            routineTimer.addTime(-fixation_cross.maxDuration)
        elif fixation_cross.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "recognition_trial" ---
        # create an object to store info about Routine recognition_trial
        recognition_trial = data.Routine(
            name='recognition_trial',
            components=[opt1, opt2, opt3, opt4, rec_key, var_1, var_2, var_3, var_4],
        )
        recognition_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        opt1.setImage(option1_img)
        opt2.setImage(option2_img)
        opt3.setImage(option3_img)
        opt4.setImage(option4_img)
        # create starting attributes for rec_key
        rec_key.keys = []
        rec_key.rt = []
        _rec_key_allKeys = []
        # store start times for recognition_trial
        recognition_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recognition_trial.tStart = globalClock.getTime(format='float')
        recognition_trial.status = STARTED
        thisExp.addData('recognition_trial.started', recognition_trial.tStart)
        recognition_trial.maxDuration = None
        # keep track of which components have finished
        recognition_trialComponents = recognition_trial.components
        for thisComponent in recognition_trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recognition_trial" ---
        thisExp.currentRoutine = recognition_trial
        recognition_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRecognition_loop, 'status') and thisRecognition_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *opt1* updates
            
            # if opt1 is starting this frame...
            if opt1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                opt1.frameNStart = frameN  # exact frame index
                opt1.tStart = t  # local t and not account for scr refresh
                opt1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(opt1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'opt1.started')
                # update status
                opt1.status = STARTED
                opt1.setAutoDraw(True)
            
            # if opt1 is active this frame...
            if opt1.status == STARTED:
                # update params
                pass
            
            # *opt2* updates
            
            # if opt2 is starting this frame...
            if opt2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                opt2.frameNStart = frameN  # exact frame index
                opt2.tStart = t  # local t and not account for scr refresh
                opt2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(opt2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'opt2.started')
                # update status
                opt2.status = STARTED
                opt2.setAutoDraw(True)
            
            # if opt2 is active this frame...
            if opt2.status == STARTED:
                # update params
                pass
            
            # *opt3* updates
            
            # if opt3 is starting this frame...
            if opt3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                opt3.frameNStart = frameN  # exact frame index
                opt3.tStart = t  # local t and not account for scr refresh
                opt3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(opt3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'opt3.started')
                # update status
                opt3.status = STARTED
                opt3.setAutoDraw(True)
            
            # if opt3 is active this frame...
            if opt3.status == STARTED:
                # update params
                pass
            
            # *opt4* updates
            
            # if opt4 is starting this frame...
            if opt4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                opt4.frameNStart = frameN  # exact frame index
                opt4.tStart = t  # local t and not account for scr refresh
                opt4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(opt4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'opt4.started')
                # update status
                opt4.status = STARTED
                opt4.setAutoDraw(True)
            
            # if opt4 is active this frame...
            if opt4.status == STARTED:
                # update params
                pass
            
            # *rec_key* updates
            waitOnFlip = False
            
            # if rec_key is starting this frame...
            if rec_key.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                rec_key.frameNStart = frameN  # exact frame index
                rec_key.tStart = t  # local t and not account for scr refresh
                rec_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rec_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rec_key.started')
                # update status
                rec_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(rec_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(rec_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if rec_key.status == STARTED and not waitOnFlip:
                theseKeys = rec_key.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                _rec_key_allKeys.extend(theseKeys)
                if len(_rec_key_allKeys):
                    rec_key.keys = _rec_key_allKeys[0].name  # just the first key pressed
                    rec_key.rt = _rec_key_allKeys[0].rt
                    rec_key.duration = _rec_key_allKeys[0].duration
                    # was this correct?
                    if (rec_key.keys == str(str(int(correct_key)))) or (rec_key.keys == str(int(correct_key))):
                        rec_key.corr = 1
                    else:
                        rec_key.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *var_1* updates
            
            # if var_1 is starting this frame...
            if var_1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                var_1.frameNStart = frameN  # exact frame index
                var_1.tStart = t  # local t and not account for scr refresh
                var_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(var_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'var_1.started')
                # update status
                var_1.status = STARTED
                var_1.setAutoDraw(True)
            
            # if var_1 is active this frame...
            if var_1.status == STARTED:
                # update params
                pass
            
            # *var_2* updates
            
            # if var_2 is starting this frame...
            if var_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                var_2.frameNStart = frameN  # exact frame index
                var_2.tStart = t  # local t and not account for scr refresh
                var_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(var_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'var_2.started')
                # update status
                var_2.status = STARTED
                var_2.setAutoDraw(True)
            
            # if var_2 is active this frame...
            if var_2.status == STARTED:
                # update params
                pass
            
            # *var_3* updates
            
            # if var_3 is starting this frame...
            if var_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                var_3.frameNStart = frameN  # exact frame index
                var_3.tStart = t  # local t and not account for scr refresh
                var_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(var_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'var_3.started')
                # update status
                var_3.status = STARTED
                var_3.setAutoDraw(True)
            
            # if var_3 is active this frame...
            if var_3.status == STARTED:
                # update params
                pass
            
            # *var_4* updates
            
            # if var_4 is starting this frame...
            if var_4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                var_4.frameNStart = frameN  # exact frame index
                var_4.tStart = t  # local t and not account for scr refresh
                var_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(var_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'var_4.started')
                # update status
                var_4.status = STARTED
                var_4.setAutoDraw(True)
            
            # if var_4 is active this frame...
            if var_4.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=recognition_trial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                recognition_trial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if recognition_trial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in recognition_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recognition_trial" ---
        for thisComponent in recognition_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recognition_trial
        recognition_trial.tStop = globalClock.getTime(format='float')
        recognition_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('recognition_trial.stopped', recognition_trial.tStop)
        # check responses
        if rec_key.keys in ['', [], None]:  # No response was made
            rec_key.keys = None
            # was no response the correct answer?!
            if str(str(int(correct_key))).lower() == 'none':
               rec_key.corr = 1;  # correct non-response
            else:
               rec_key.corr = 0;  # failed to respond (incorrectly)
        # store data for recognition_loop (TrialHandler)
        recognition_loop.addData('rec_key.keys',rec_key.keys)
        recognition_loop.addData('rec_key.corr', rec_key.corr)
        if rec_key.keys != None:  # we had a response
            recognition_loop.addData('rec_key.rt', rec_key.rt)
            recognition_loop.addData('rec_key.duration', rec_key.duration)
        # the Routine "recognition_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisRecognition_loop as finished
        if hasattr(thisRecognition_loop, 'status'):
            thisRecognition_loop.status = FINISHED
        # if awaiting a pause, pause now
        if recognition_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            recognition_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'recognition_loop'
    recognition_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "End_practice1" ---
    # create an object to store info about Routine End_practice1
    End_practice1 = data.Routine(
        name='End_practice1',
        components=[end_practice1, mouse1],
    )
    End_practice1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse1
    mouse1.x = []
    mouse1.y = []
    mouse1.leftButton = []
    mouse1.midButton = []
    mouse1.rightButton = []
    mouse1.time = []
    gotValidClick = False  # until a click is received
    # store start times for End_practice1
    End_practice1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    End_practice1.tStart = globalClock.getTime(format='float')
    End_practice1.status = STARTED
    thisExp.addData('End_practice1.started', End_practice1.tStart)
    End_practice1.maxDuration = None
    # keep track of which components have finished
    End_practice1Components = End_practice1.components
    for thisComponent in End_practice1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "End_practice1" ---
    thisExp.currentRoutine = End_practice1
    End_practice1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_practice1* updates
        
        # if end_practice1 is starting this frame...
        if end_practice1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            end_practice1.frameNStart = frameN  # exact frame index
            end_practice1.tStart = t  # local t and not account for scr refresh
            end_practice1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_practice1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_practice1.started')
            # update status
            end_practice1.status = STARTED
            end_practice1.setAutoDraw(True)
        
        # if end_practice1 is active this frame...
        if end_practice1.status == STARTED:
            # update params
            pass
        # *mouse1* updates
        
        # if mouse1 is starting this frame...
        if mouse1.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            mouse1.frameNStart = frameN  # exact frame index
            mouse1.tStart = t  # local t and not account for scr refresh
            mouse1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse1.started', t)
            # update status
            mouse1.status = STARTED
            mouse1.mouseClock.reset()
            # if button is down already this ISN'T a new click
            mouse1.prevButtonState = mouse1.getPressed()
        if mouse1.status == STARTED:  # only update if started and not finished!
            buttons = mouse1.getPressed()
            # button state changed?
            if buttons != mouse1.prevButtonState:
                mouse1.prevButtonState = buttons
                # state changed to a new click
                if sum(buttons) > 0:
                    pass
                    x, y = mouse1.getPos()
                    mouse1.x.append(float(x))
                    mouse1.y.append(float(y))
                    buttons = mouse1.getPressed()
                    mouse1.leftButton.append(buttons[0])
                    mouse1.midButton.append(buttons[1])
                    mouse1.rightButton.append(buttons[2])
                    mouse1.time.append(mouse1.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=End_practice1,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            End_practice1.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if End_practice1.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in End_practice1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End_practice1" ---
    for thisComponent in End_practice1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for End_practice1
    End_practice1.tStop = globalClock.getTime(format='float')
    End_practice1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('End_practice1.stopped', End_practice1.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse1.x', mouse1.x)
    thisExp.addData('mouse1.y', mouse1.y)
    thisExp.addData('mouse1.leftButton', mouse1.leftButton)
    thisExp.addData('mouse1.midButton', mouse1.midButton)
    thisExp.addData('mouse1.rightButton', mouse1.rightButton)
    thisExp.addData('mouse1.time', mouse1.time)
    thisExp.nextEntry()
    # the Routine "End_practice1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
