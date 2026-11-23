#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.2.4),
    on November 23, 2026, at 12:38
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
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware, iohub
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
from psychopy.hardware.eyetracker import EyetrackerCalibration
from psychopy.hardware.eyetracker import EyetrackerControl

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
    'participant': '00',
    'age': '00',
    'gender': ['male', 'female', 'diverse', 'prefer not to say'],
    'handedness': ['right', 'left', 'ambidextrous'],
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
_winSize = [1680,1050]
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
        originPath='D:\\Master\\Psychophysics\\code\\Face-like-Configural-Organization-Attentional-Selection-and-Concurrent-Scene-Memory\\Raiden_experimenteye.py',
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
    # setup eyetracking
    ioConfig['eyetracker.hw.sr_research.eyelink.EyeTracker'] = {
        'name': 'tracker',
    }
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    deviceManager.devices['eyetracker'] = ioServer.getDevice('tracker')
    
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
    # Run 'Begin Experiment' code from important_setups
    import trial_builder 
     
    # Get participant ID 
    participant = str(expInfo['participant']) 
     
    # Generate trial lists 
    trial_info = trial_builder.build_all( 
        participant, 
        write=True 
    ) 
     
    sequence = trial_info['sequence'] 
    settings = trial_info['settings'] 
     
    # Participant folder 
    participantFolder = 'trial_lists/' + participant 
     
    sessionFile = ( 
        participantFolder + 
        '/P' + participant + 
        '_session_summary.csv' 
    ) 
     
    print("Participant:", participant) 
    print("Sequence:", sequence) 
    print("Settings:", settings)
    
    # --- Initialize components for Routine "startrecording" ---
    etRecord = EyetrackerControl(
        tracker=eyetracker,
        actionType='Start Only'
    )
    
    # --- Initialize components for Routine "Instructions_for_practice" ---
    instr_text = visual.TextStim(win=win, name='instr_text',
        text='',
        font='Arial',
        pos=[0, 0.3], draggable=False, height=0.035, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    intr_key = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    label_nonface = visual.TextStim(win=win, name='label_nonface',
        text='NON-FACE',
        font='Arial',
        pos=[0.3, -0.02], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    label_face = visual.TextStim(win=win, name='label_face',
        text='FACE',
        font='Arial',
        pos=[-0.3, -0.02], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    ex_nonface = visual.ImageStim(
        win=win,
        name='ex_nonface', 
        image='practice/non_target_1.png', mask=None, anchor='center',
        ori=0.0, pos=[0.3, -0.22], draggable=False, size=[0.3, 0.3],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    ex_face = visual.ImageStim(
        win=win,
        name='ex_face', 
        image='practice/target_1.png', mask=None, anchor='center',
        ori=0.0, pos=[-0.3, -0.22], draggable=False, size=[0.3, 0.3],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    
    # --- Initialize components for Routine "fixation_cross" ---
    fixation_cross_display = visual.TextStim(win=win, name='fixation_cross_display',
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
    
    # --- Initialize components for Routine "Instruction" ---
    inst_text = visual.TextStim(win=win, name='inst_text',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intr_key1 = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    
    # --- Initialize components for Routine "fixation_cross2" ---
    fixation_cross_display2 = visual.TextStim(win=win, name='fixation_cross_display2',
        text='+',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "encoding_trial" ---
    Background_Encoding = visual.ImageStim(
        win=win,
        name='Background_Encoding', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0, 0], draggable=False, size=[0.8, 0.8],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    Central_Encoding = visual.ImageStim(
        win=win,
        name='Central_Encoding', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0, 0], draggable=False, size=[0.15, 0.15],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    response_encoding = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    
    # --- Initialize components for Routine "stoprecord" ---
    etRecord_stop = EyetrackerControl(
        tracker=eyetracker,
        actionType='Stop Only'
    )
    
    # --- Initialize components for Routine "inst_recognition" ---
    inst_recognition_text = visual.TextStim(win=win, name='inst_recognition_text',
        text="Please select which image you just saw by pressing the corresponding numbers.\n\nPress 'SPACE' to continue",
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    fix1 = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    
    # --- Initialize components for Routine "fixation_cross3" ---
    fixation_cross_display3 = visual.TextStim(win=win, name='fixation_cross_display3',
        text='+',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "recognition_trial" ---
    opt5 = visual.ImageStim(
        win=win,
        name='opt5', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[-0.6, 0], draggable=False, size=[0.36, 0.36],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    opt5_2 = visual.ImageStim(
        win=win,
        name='opt5_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[-0.2, 0], draggable=False, size=[0.36, 0.36],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    opt5_3 = visual.ImageStim(
        win=win,
        name='opt5_3', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0.2, 0], draggable=False, size=[0.36, 0.36],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    opt5_4 = visual.ImageStim(
        win=win,
        name='opt5_4', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0.6, 0], draggable=False, size=[0.36, 0.36],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    rec_key1 = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    var_5 = visual.TextStim(win=win, name='var_5',
        text='1',
        font='Arial',
        pos=[-0.6, -0.25], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    var = visual.TextStim(win=win, name='var',
        text='2',
        font='Arial',
        pos=[-0.2, -0.25], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    var_6 = visual.TextStim(win=win, name='var_6',
        text='3',
        font='Arial',
        pos=[0.2, -0.25], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    var_7 = visual.TextStim(win=win, name='var_7',
        text='4',
        font='Arial',
        pos=[0.6, -0.25], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    
    # --- Initialize components for Routine "show_end_block" ---
    end_block_text = visual.TextStim(win=win, name='end_block_text',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    end_block = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    
    # --- Initialize components for Routine "goodbye" ---
    goodbye_text = visual.TextStim(win=win, name='goodbye_text',
        text='Thank you very much for participating our experiment!\n\nPlease inform the experimenter.\n\nGoodbye!',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    exit_exp = keyboard.Keyboard(deviceName='defaultKeyboard', backend='ioHub')
    
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
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler2(
        name='blocks',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(sessionFile), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(blocks)  # add the loop to the experiment
    thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlock in blocks:
        blocks.status = STARTED
        if hasattr(thisBlock, 'status'):
            thisBlock.status = STARTED
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        # define target for calibration
        calibrationTarget = visual.TargetStim(win, 
            name='calibrationTarget',
            radius=0.01, 
            fillColor='', 
            borderColor='black', 
            lineWidth=2.0, 
            innerRadius=0.0035, 
            innerFillColor='green', 
            innerBorderColor='black', 
            innerLineWidth=2.0,
            colorSpace='rgb', 
            units=None
        )
        # define parameters for calibration
        calibration = EyetrackerCalibration(win, 
            eyetracker, calibrationTarget,
            units=None, colorSpace='rgb',
            progressMode='time', targetDur=1.5, expandScale=1.5,
            targetLayout='NINE_POINTS', randomisePos=True, textColor='white',
            movementAnimation=True, targetDelay=1.0
        )
        # run calibration
        calibration.run()
        # clear any keypresses from during calibration so they don't interfere with the experiment
        defaultKeyboard.clearEvents()
        # the Routine "calibration" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "startrecording" ---
        # create an object to store info about Routine startrecording
        startrecording = data.Routine(
            name='startrecording',
            components=[etRecord],
        )
        startrecording.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for startrecording
        startrecording.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        startrecording.tStart = globalClock.getTime(format='float')
        startrecording.status = STARTED
        thisExp.addData('startrecording.started', startrecording.tStart)
        startrecording.maxDuration = None
        # keep track of which components have finished
        startrecordingComponents = startrecording.components
        for thisComponent in startrecording.components:
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
        
        # --- Run Routine "startrecording" ---
        thisExp.currentRoutine = startrecording
        startrecording.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock, 'status') and thisBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *etRecord* updates
            
            # if etRecord is starting this frame...
            if etRecord.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                etRecord.frameNStart = frameN  # exact frame index
                etRecord.tStart = t  # local t and not account for scr refresh
                etRecord.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(etRecord, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('etRecord.started', t)
                # update status
                etRecord.status = STARTED
                etRecord.start()
            if etRecord.status == STARTED:
                etRecord.tStop = t  # not accounting for scr refresh
                etRecord.tStopRefresh = tThisFlipGlobal  # on global time
                etRecord.frameNStop = frameN  # exact frame index
                etRecord.status = FINISHED
            
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
                    currentRoutine=startrecording,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                startrecording.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if startrecording.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in startrecording.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "startrecording" ---
        for thisComponent in startrecording.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for startrecording
        startrecording.tStop = globalClock.getTime(format='float')
        startrecording.tStopRefresh = tThisFlipGlobal
        thisExp.addData('startrecording.stopped', startrecording.tStop)
        # the Routine "startrecording" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Instructions_for_practice" ---
        # create an object to store info about Routine Instructions_for_practice
        Instructions_for_practice = data.Routine(
            name='Instructions_for_practice',
            components=[instr_text, intr_key, label_nonface, label_face, ex_nonface, ex_face],
        )
        Instructions_for_practice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from session_setup
        sessionNum = blocks.thisN + 1 
         
        encodingFile = ( 
            participantFolder + 
            '/P' + participant + 
            '_S' + str(sessionNum) + 
            '_encoding.csv' 
        ) 
         
        recognitionFile = ( 
            participantFolder + 
            '/P' + participant + 
            '_S' + str(sessionNum) + 
            '_recognition.csv' 
        ) 
         
        print("Session:", sessionNum) 
        print("Encoding:", encodingFile) 
        print("Recognition:", recognitionFile)
        instr_text.setText("Block "+ str(sessionNum) + "/4 \n\n In this experiment, you will press SPACE when you see the target. /n/n The target for this block is " + target + " images. \n\n Examples of a target and a non-target are shown below. \n\n To start the practice, please press SPACE.")
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
            # if trial has changed, end Routine now
            if hasattr(thisBlock, 'status') and thisBlock.status == STOPPING:
                continueRoutine = False
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
            
            # *label_nonface* updates
            
            # if label_nonface is starting this frame...
            if label_nonface.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                label_nonface.frameNStart = frameN  # exact frame index
                label_nonface.tStart = t  # local t and not account for scr refresh
                label_nonface.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(label_nonface, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'label_nonface.started')
                # update status
                label_nonface.status = STARTED
                label_nonface.setAutoDraw(True)
            
            # if label_nonface is active this frame...
            if label_nonface.status == STARTED:
                # update params
                pass
            
            # *label_face* updates
            
            # if label_face is starting this frame...
            if label_face.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                label_face.frameNStart = frameN  # exact frame index
                label_face.tStart = t  # local t and not account for scr refresh
                label_face.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(label_face, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'label_face.started')
                # update status
                label_face.status = STARTED
                label_face.setAutoDraw(True)
            
            # if label_face is active this frame...
            if label_face.status == STARTED:
                # update params
                pass
            
            # *ex_nonface* updates
            
            # if ex_nonface is starting this frame...
            if ex_nonface.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                ex_nonface.frameNStart = frameN  # exact frame index
                ex_nonface.tStart = t  # local t and not account for scr refresh
                ex_nonface.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ex_nonface, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ex_nonface.started')
                # update status
                ex_nonface.status = STARTED
                ex_nonface.setAutoDraw(True)
            
            # if ex_nonface is active this frame...
            if ex_nonface.status == STARTED:
                # update params
                pass
            
            # *ex_face* updates
            
            # if ex_face is starting this frame...
            if ex_face.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                ex_face.frameNStart = frameN  # exact frame index
                ex_face.tStart = t  # local t and not account for scr refresh
                ex_face.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ex_face, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ex_face.started')
                # update status
                ex_face.status = STARTED
                ex_face.setAutoDraw(True)
            
            # if ex_face is active this frame...
            if ex_face.status == STARTED:
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
        blocks.addData('intr_key.keys',intr_key.keys)
        if intr_key.keys != None:  # we had a response
            blocks.addData('intr_key.rt', intr_key.rt)
            blocks.addData('intr_key.duration', intr_key.duration)
        # the Routine "Instructions_for_practice" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('practice_trial.xlsx'), 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            trials.status = STARTED
            if hasattr(thisTrial, 'status'):
                thisTrial.status = STARTED
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "fixation_cross" ---
            # create an object to store info about Routine fixation_cross
            fixation_cross = data.Routine(
                name='fixation_cross',
                components=[fixation_cross_display],
            )
            fixation_cross.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_2
            ioServer.sendMessageEvent('Fixation t%i')
            eyetracker.sendMessage('Fixation t%i')
            
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
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixation_cross_display* updates
                
                # if fixation_cross_display is starting this frame...
                if fixation_cross_display.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    fixation_cross_display.frameNStart = frameN  # exact frame index
                    fixation_cross_display.tStart = t  # local t and not account for scr refresh
                    fixation_cross_display.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixation_cross_display, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_cross_display.started')
                    # update status
                    fixation_cross_display.status = STARTED
                    fixation_cross_display.setAutoDraw(True)
                
                # if fixation_cross_display is active this frame...
                if fixation_cross_display.status == STARTED:
                    # update params
                    pass
                
                # if fixation_cross_display is stopping this frame...
                if fixation_cross_display.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixation_cross_display.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        fixation_cross_display.tStop = t  # not accounting for scr refresh
                        fixation_cross_display.tStopRefresh = tThisFlipGlobal  # on global time
                        fixation_cross_display.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixation_cross_display.stopped')
                        # update status
                        fixation_cross_display.status = FINISHED
                        fixation_cross_display.setAutoDraw(False)
                
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
            # Run 'Begin Routine' code from code
            ioServer.sendMessageEvent('Search begin t%i')
            eyetracker.sendMessage('Search begin t%i')
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
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
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
                        if (response.keys == str(correct_key)) or (response.keys == correct_key):
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
                if str(correct_key).lower() == 'none':
                   response.corr = 1;  # correct non-response
                else:
                   response.corr = 0;  # failed to respond (incorrectly)
            # store data for trials (TrialHandler)
            trials.addData('response.keys',response.keys)
            trials.addData('response.corr', response.corr)
            if response.keys != None:  # we had a response
                trials.addData('response.rt', response.rt)
                trials.addData('response.duration', response.duration)
            # Run 'End Routine' code from code
            ioServer.sendMessageEvent('Search end t%i')
            eyetracker.sendMessage('Search end t%i')
            # the Routine "Practice_Trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisTrial as finished
            if hasattr(thisTrial, 'status'):
                thisTrial.status = FINISHED
            # if awaiting a pause, pause now
            if trials.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                trials.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials'
        trials.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "Instruction" ---
        # create an object to store info about Routine Instruction
        Instruction = data.Routine(
            name='Instruction',
            components=[inst_text, intr_key1],
        )
        Instruction.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        inst_text.setText("Block " + str(sessionNum) + "/ 4 \n\n In this experiment, you will press 'SPACE' when you see the target. \n\n The target for this block is " + target + " images same as the practice trial.\n\n Please memorize the background images, later you will be probed about background images. \n\n To start the block, please press SPACE.")
        # create starting attributes for intr_key1
        intr_key1.keys = []
        intr_key1.rt = []
        _intr_key1_allKeys = []
        # store start times for Instruction
        Instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Instruction.tStart = globalClock.getTime(format='float')
        Instruction.status = STARTED
        thisExp.addData('Instruction.started', Instruction.tStart)
        Instruction.maxDuration = None
        # keep track of which components have finished
        InstructionComponents = Instruction.components
        for thisComponent in Instruction.components:
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
        
        # --- Run Routine "Instruction" ---
        thisExp.currentRoutine = Instruction
        Instruction.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock, 'status') and thisBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *inst_text* updates
            
            # if inst_text is starting this frame...
            if inst_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                inst_text.frameNStart = frameN  # exact frame index
                inst_text.tStart = t  # local t and not account for scr refresh
                inst_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(inst_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'inst_text.started')
                # update status
                inst_text.status = STARTED
                inst_text.setAutoDraw(True)
            
            # if inst_text is active this frame...
            if inst_text.status == STARTED:
                # update params
                pass
            
            # *intr_key1* updates
            waitOnFlip = False
            
            # if intr_key1 is starting this frame...
            if intr_key1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                intr_key1.frameNStart = frameN  # exact frame index
                intr_key1.tStart = t  # local t and not account for scr refresh
                intr_key1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(intr_key1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'intr_key1.started')
                # update status
                intr_key1.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(intr_key1.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(intr_key1.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if intr_key1.status == STARTED and not waitOnFlip:
                theseKeys = intr_key1.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _intr_key1_allKeys.extend(theseKeys)
                if len(_intr_key1_allKeys):
                    intr_key1.keys = _intr_key1_allKeys[-1].name  # just the last key pressed
                    intr_key1.rt = _intr_key1_allKeys[-1].rt
                    intr_key1.duration = _intr_key1_allKeys[-1].duration
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
                    currentRoutine=Instruction,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Instruction.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Instruction.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Instruction.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Instruction" ---
        for thisComponent in Instruction.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Instruction
        Instruction.tStop = globalClock.getTime(format='float')
        Instruction.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Instruction.stopped', Instruction.tStop)
        # check responses
        if intr_key1.keys in ['', [], None]:  # No response was made
            intr_key1.keys = None
        blocks.addData('intr_key1.keys',intr_key1.keys)
        if intr_key1.keys != None:  # we had a response
            blocks.addData('intr_key1.rt', intr_key1.rt)
            blocks.addData('intr_key1.duration', intr_key1.duration)
        # the Routine "Instruction" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        encoding_trials = data.TrialHandler2(
            name='encoding_trials',
            nReps=1, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(encodingFile), 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(encoding_trials)  # add the loop to the experiment
        thisEncoding_trial = encoding_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisEncoding_trial.rgb)
        if thisEncoding_trial != None:
            for paramName in thisEncoding_trial:
                globals()[paramName] = thisEncoding_trial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisEncoding_trial in encoding_trials:
            encoding_trials.status = STARTED
            if hasattr(thisEncoding_trial, 'status'):
                thisEncoding_trial.status = STARTED
            currentLoop = encoding_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisEncoding_trial.rgb)
            if thisEncoding_trial != None:
                for paramName in thisEncoding_trial:
                    globals()[paramName] = thisEncoding_trial[paramName]
            
            # --- Prepare to start Routine "fixation_cross2" ---
            # create an object to store info about Routine fixation_cross2
            fixation_cross2 = data.Routine(
                name='fixation_cross2',
                components=[fixation_cross_display2],
            )
            fixation_cross2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for fixation_cross2
            fixation_cross2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            fixation_cross2.tStart = globalClock.getTime(format='float')
            fixation_cross2.status = STARTED
            thisExp.addData('fixation_cross2.started', fixation_cross2.tStart)
            fixation_cross2.maxDuration = None
            # keep track of which components have finished
            fixation_cross2Components = fixation_cross2.components
            for thisComponent in fixation_cross2.components:
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
            
            # --- Run Routine "fixation_cross2" ---
            thisExp.currentRoutine = fixation_cross2
            fixation_cross2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.5:
                # if trial has changed, end Routine now
                if hasattr(thisEncoding_trial, 'status') and thisEncoding_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixation_cross_display2* updates
                
                # if fixation_cross_display2 is starting this frame...
                if fixation_cross_display2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    fixation_cross_display2.frameNStart = frameN  # exact frame index
                    fixation_cross_display2.tStart = t  # local t and not account for scr refresh
                    fixation_cross_display2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixation_cross_display2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_cross_display2.started')
                    # update status
                    fixation_cross_display2.status = STARTED
                    fixation_cross_display2.setAutoDraw(True)
                
                # if fixation_cross_display2 is active this frame...
                if fixation_cross_display2.status == STARTED:
                    # update params
                    pass
                
                # if fixation_cross_display2 is stopping this frame...
                if fixation_cross_display2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixation_cross_display2.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        fixation_cross_display2.tStop = t  # not accounting for scr refresh
                        fixation_cross_display2.tStopRefresh = tThisFlipGlobal  # on global time
                        fixation_cross_display2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixation_cross_display2.stopped')
                        # update status
                        fixation_cross_display2.status = FINISHED
                        fixation_cross_display2.setAutoDraw(False)
                
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
                        currentRoutine=fixation_cross2,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    fixation_cross2.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if fixation_cross2.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in fixation_cross2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation_cross2" ---
            for thisComponent in fixation_cross2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for fixation_cross2
            fixation_cross2.tStop = globalClock.getTime(format='float')
            fixation_cross2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('fixation_cross2.stopped', fixation_cross2.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if fixation_cross2.maxDurationReached:
                routineTimer.addTime(-fixation_cross2.maxDuration)
            elif fixation_cross2.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
            
            # --- Prepare to start Routine "encoding_trial" ---
            # create an object to store info about Routine encoding_trial
            encoding_trial = data.Routine(
                name='encoding_trial',
                components=[Background_Encoding, Central_Encoding, response_encoding],
            )
            encoding_trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            Background_Encoding.setImage(background_img)
            Central_Encoding.setImage(central_img)
            # create starting attributes for response_encoding
            response_encoding.keys = []
            response_encoding.rt = []
            _response_encoding_allKeys = []
            # store start times for encoding_trial
            encoding_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            encoding_trial.tStart = globalClock.getTime(format='float')
            encoding_trial.status = STARTED
            thisExp.addData('encoding_trial.started', encoding_trial.tStart)
            encoding_trial.maxDuration = None
            # keep track of which components have finished
            encoding_trialComponents = encoding_trial.components
            for thisComponent in encoding_trial.components:
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
            
            # --- Run Routine "encoding_trial" ---
            thisExp.currentRoutine = encoding_trial
            encoding_trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisEncoding_trial, 'status') and thisEncoding_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *Background_Encoding* updates
                
                # if Background_Encoding is starting this frame...
                if Background_Encoding.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    Background_Encoding.frameNStart = frameN  # exact frame index
                    Background_Encoding.tStart = t  # local t and not account for scr refresh
                    Background_Encoding.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Background_Encoding, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Background_Encoding.started')
                    # update status
                    Background_Encoding.status = STARTED
                    Background_Encoding.setAutoDraw(True)
                
                # if Background_Encoding is active this frame...
                if Background_Encoding.status == STARTED:
                    # update params
                    pass
                
                # if Background_Encoding is stopping this frame...
                if Background_Encoding.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Background_Encoding.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        Background_Encoding.tStop = t  # not accounting for scr refresh
                        Background_Encoding.tStopRefresh = tThisFlipGlobal  # on global time
                        Background_Encoding.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Background_Encoding.stopped')
                        # update status
                        Background_Encoding.status = FINISHED
                        Background_Encoding.setAutoDraw(False)
                
                # *Central_Encoding* updates
                
                # if Central_Encoding is starting this frame...
                if Central_Encoding.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    Central_Encoding.frameNStart = frameN  # exact frame index
                    Central_Encoding.tStart = t  # local t and not account for scr refresh
                    Central_Encoding.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Central_Encoding, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Central_Encoding.started')
                    # update status
                    Central_Encoding.status = STARTED
                    Central_Encoding.setAutoDraw(True)
                
                # if Central_Encoding is active this frame...
                if Central_Encoding.status == STARTED:
                    # update params
                    pass
                
                # if Central_Encoding is stopping this frame...
                if Central_Encoding.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Central_Encoding.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        Central_Encoding.tStop = t  # not accounting for scr refresh
                        Central_Encoding.tStopRefresh = tThisFlipGlobal  # on global time
                        Central_Encoding.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Central_Encoding.stopped')
                        # update status
                        Central_Encoding.status = FINISHED
                        Central_Encoding.setAutoDraw(False)
                
                # *response_encoding* updates
                waitOnFlip = False
                
                # if response_encoding is starting this frame...
                if response_encoding.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    response_encoding.frameNStart = frameN  # exact frame index
                    response_encoding.tStart = t  # local t and not account for scr refresh
                    response_encoding.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(response_encoding, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'response_encoding.started')
                    # update status
                    response_encoding.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(response_encoding.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(response_encoding.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if response_encoding.status == STARTED and not waitOnFlip:
                    theseKeys = response_encoding.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _response_encoding_allKeys.extend(theseKeys)
                    if len(_response_encoding_allKeys):
                        response_encoding.keys = _response_encoding_allKeys[-1].name  # just the last key pressed
                        response_encoding.rt = _response_encoding_allKeys[-1].rt
                        response_encoding.duration = _response_encoding_allKeys[-1].duration
                        # was this correct?
                        if (response_encoding.keys == str(correct_key)) or (response_encoding.keys == correct_key):
                            response_encoding.corr = 1
                        else:
                            response_encoding.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                # Run 'Each Frame' code from code_encoding
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
                        currentRoutine=encoding_trial,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    encoding_trial.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if encoding_trial.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in encoding_trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "encoding_trial" ---
            for thisComponent in encoding_trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for encoding_trial
            encoding_trial.tStop = globalClock.getTime(format='float')
            encoding_trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('encoding_trial.stopped', encoding_trial.tStop)
            # check responses
            if response_encoding.keys in ['', [], None]:  # No response was made
                response_encoding.keys = None
                # was no response the correct answer?!
                if str(correct_key).lower() == 'none':
                   response_encoding.corr = 1;  # correct non-response
                else:
                   response_encoding.corr = 0;  # failed to respond (incorrectly)
            # store data for encoding_trials (TrialHandler)
            encoding_trials.addData('response_encoding.keys',response_encoding.keys)
            encoding_trials.addData('response_encoding.corr', response_encoding.corr)
            if response_encoding.keys != None:  # we had a response
                encoding_trials.addData('response_encoding.rt', response_encoding.rt)
                encoding_trials.addData('response_encoding.duration', response_encoding.duration)
            # the Routine "encoding_trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisEncoding_trial as finished
            if hasattr(thisEncoding_trial, 'status'):
                thisEncoding_trial.status = FINISHED
            # if awaiting a pause, pause now
            if encoding_trials.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                encoding_trials.status = STARTED
            thisExp.nextEntry()
            
        # completed 1 repeats of 'encoding_trials'
        encoding_trials.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "stoprecord" ---
        # create an object to store info about Routine stoprecord
        stoprecord = data.Routine(
            name='stoprecord',
            components=[etRecord_stop],
        )
        stoprecord.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for stoprecord
        stoprecord.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        stoprecord.tStart = globalClock.getTime(format='float')
        stoprecord.status = STARTED
        thisExp.addData('stoprecord.started', stoprecord.tStart)
        stoprecord.maxDuration = None
        # keep track of which components have finished
        stoprecordComponents = stoprecord.components
        for thisComponent in stoprecord.components:
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
        
        # --- Run Routine "stoprecord" ---
        thisExp.currentRoutine = stoprecord
        stoprecord.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisBlock, 'status') and thisBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *etRecord_stop* updates
            if etRecord_stop.status == NOT_STARTED:
                etRecord_stop.frameNStart = frameN  # exact frame index
                etRecord_stop.tStart = t  # local t and not account for scr refresh
                etRecord_stop.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(etRecord_stop, 'tStartRefresh')  # time at next scr refresh
                etRecord_stop.status = STARTED
            
            # if etRecord_stop is stopping this frame...
            if etRecord_stop.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > etRecord_stop.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    etRecord_stop.tStop = t  # not accounting for scr refresh
                    etRecord_stop.tStopRefresh = tThisFlipGlobal  # on global time
                    etRecord_stop.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('etRecord_stop.stopped', t)
                    # update status
                    etRecord_stop.status = FINISHED
                    etRecord_stop.stop()
            
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
                    currentRoutine=stoprecord,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                stoprecord.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if stoprecord.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in stoprecord.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "stoprecord" ---
        for thisComponent in stoprecord.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for stoprecord
        stoprecord.tStop = globalClock.getTime(format='float')
        stoprecord.tStopRefresh = tThisFlipGlobal
        thisExp.addData('stoprecord.stopped', stoprecord.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if stoprecord.maxDurationReached:
            routineTimer.addTime(-stoprecord.maxDuration)
        elif stoprecord.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "inst_recognition" ---
        # create an object to store info about Routine inst_recognition
        inst_recognition = data.Routine(
            name='inst_recognition',
            components=[inst_recognition_text, fix1],
        )
        inst_recognition.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for fix1
        fix1.keys = []
        fix1.rt = []
        _fix1_allKeys = []
        # store start times for inst_recognition
        inst_recognition.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        inst_recognition.tStart = globalClock.getTime(format='float')
        inst_recognition.status = STARTED
        thisExp.addData('inst_recognition.started', inst_recognition.tStart)
        inst_recognition.maxDuration = None
        # keep track of which components have finished
        inst_recognitionComponents = inst_recognition.components
        for thisComponent in inst_recognition.components:
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
        
        # --- Run Routine "inst_recognition" ---
        thisExp.currentRoutine = inst_recognition
        inst_recognition.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock, 'status') and thisBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *inst_recognition_text* updates
            
            # if inst_recognition_text is starting this frame...
            if inst_recognition_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                inst_recognition_text.frameNStart = frameN  # exact frame index
                inst_recognition_text.tStart = t  # local t and not account for scr refresh
                inst_recognition_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(inst_recognition_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'inst_recognition_text.started')
                # update status
                inst_recognition_text.status = STARTED
                inst_recognition_text.setAutoDraw(True)
            
            # if inst_recognition_text is active this frame...
            if inst_recognition_text.status == STARTED:
                # update params
                pass
            
            # *fix1* updates
            waitOnFlip = False
            
            # if fix1 is starting this frame...
            if fix1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                fix1.frameNStart = frameN  # exact frame index
                fix1.tStart = t  # local t and not account for scr refresh
                fix1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix1.started')
                # update status
                fix1.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(fix1.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(fix1.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if fix1.status == STARTED and not waitOnFlip:
                theseKeys = fix1.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _fix1_allKeys.extend(theseKeys)
                if len(_fix1_allKeys):
                    fix1.keys = _fix1_allKeys[-1].name  # just the last key pressed
                    fix1.rt = _fix1_allKeys[-1].rt
                    fix1.duration = _fix1_allKeys[-1].duration
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
                    currentRoutine=inst_recognition,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                inst_recognition.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if inst_recognition.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in inst_recognition.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "inst_recognition" ---
        for thisComponent in inst_recognition.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for inst_recognition
        inst_recognition.tStop = globalClock.getTime(format='float')
        inst_recognition.tStopRefresh = tThisFlipGlobal
        thisExp.addData('inst_recognition.stopped', inst_recognition.tStop)
        # check responses
        if fix1.keys in ['', [], None]:  # No response was made
            fix1.keys = None
        blocks.addData('fix1.keys',fix1.keys)
        if fix1.keys != None:  # we had a response
            blocks.addData('fix1.rt', fix1.rt)
            blocks.addData('fix1.duration', fix1.duration)
        # the Routine "inst_recognition" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        recognition_trials = data.TrialHandler2(
            name='recognition_trials',
            nReps=1, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(recognitionFile), 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(recognition_trials)  # add the loop to the experiment
        thisRecognition_trial = recognition_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisRecognition_trial.rgb)
        if thisRecognition_trial != None:
            for paramName in thisRecognition_trial:
                globals()[paramName] = thisRecognition_trial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisRecognition_trial in recognition_trials:
            recognition_trials.status = STARTED
            if hasattr(thisRecognition_trial, 'status'):
                thisRecognition_trial.status = STARTED
            currentLoop = recognition_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisRecognition_trial.rgb)
            if thisRecognition_trial != None:
                for paramName in thisRecognition_trial:
                    globals()[paramName] = thisRecognition_trial[paramName]
            
            # --- Prepare to start Routine "fixation_cross3" ---
            # create an object to store info about Routine fixation_cross3
            fixation_cross3 = data.Routine(
                name='fixation_cross3',
                components=[fixation_cross_display3],
            )
            fixation_cross3.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for fixation_cross3
            fixation_cross3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            fixation_cross3.tStart = globalClock.getTime(format='float')
            fixation_cross3.status = STARTED
            thisExp.addData('fixation_cross3.started', fixation_cross3.tStart)
            fixation_cross3.maxDuration = None
            # keep track of which components have finished
            fixation_cross3Components = fixation_cross3.components
            for thisComponent in fixation_cross3.components:
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
            
            # --- Run Routine "fixation_cross3" ---
            thisExp.currentRoutine = fixation_cross3
            fixation_cross3.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.5:
                # if trial has changed, end Routine now
                if hasattr(thisRecognition_trial, 'status') and thisRecognition_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixation_cross_display3* updates
                
                # if fixation_cross_display3 is starting this frame...
                if fixation_cross_display3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    fixation_cross_display3.frameNStart = frameN  # exact frame index
                    fixation_cross_display3.tStart = t  # local t and not account for scr refresh
                    fixation_cross_display3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixation_cross_display3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_cross_display3.started')
                    # update status
                    fixation_cross_display3.status = STARTED
                    fixation_cross_display3.setAutoDraw(True)
                
                # if fixation_cross_display3 is active this frame...
                if fixation_cross_display3.status == STARTED:
                    # update params
                    pass
                
                # if fixation_cross_display3 is stopping this frame...
                if fixation_cross_display3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixation_cross_display3.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        fixation_cross_display3.tStop = t  # not accounting for scr refresh
                        fixation_cross_display3.tStopRefresh = tThisFlipGlobal  # on global time
                        fixation_cross_display3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixation_cross_display3.stopped')
                        # update status
                        fixation_cross_display3.status = FINISHED
                        fixation_cross_display3.setAutoDraw(False)
                
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
                        currentRoutine=fixation_cross3,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    fixation_cross3.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if fixation_cross3.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in fixation_cross3.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation_cross3" ---
            for thisComponent in fixation_cross3.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for fixation_cross3
            fixation_cross3.tStop = globalClock.getTime(format='float')
            fixation_cross3.tStopRefresh = tThisFlipGlobal
            thisExp.addData('fixation_cross3.stopped', fixation_cross3.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if fixation_cross3.maxDurationReached:
                routineTimer.addTime(-fixation_cross3.maxDuration)
            elif fixation_cross3.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
            
            # --- Prepare to start Routine "recognition_trial" ---
            # create an object to store info about Routine recognition_trial
            recognition_trial = data.Routine(
                name='recognition_trial',
                components=[opt5, opt5_2, opt5_3, opt5_4, rec_key1, var_5, var, var_6, var_7],
            )
            recognition_trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            opt5.setImage(option1_img)
            opt5_2.setImage(option2_img)
            opt5_3.setImage(option3_img)
            opt5_4.setImage(option4_img)
            # create starting attributes for rec_key1
            rec_key1.keys = []
            rec_key1.rt = []
            _rec_key1_allKeys = []
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
                if hasattr(thisRecognition_trial, 'status') and thisRecognition_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *opt5* updates
                
                # if opt5 is starting this frame...
                if opt5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    opt5.frameNStart = frameN  # exact frame index
                    opt5.tStart = t  # local t and not account for scr refresh
                    opt5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(opt5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'opt5.started')
                    # update status
                    opt5.status = STARTED
                    opt5.setAutoDraw(True)
                
                # if opt5 is active this frame...
                if opt5.status == STARTED:
                    # update params
                    pass
                
                # *opt5_2* updates
                
                # if opt5_2 is starting this frame...
                if opt5_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    opt5_2.frameNStart = frameN  # exact frame index
                    opt5_2.tStart = t  # local t and not account for scr refresh
                    opt5_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(opt5_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'opt5_2.started')
                    # update status
                    opt5_2.status = STARTED
                    opt5_2.setAutoDraw(True)
                
                # if opt5_2 is active this frame...
                if opt5_2.status == STARTED:
                    # update params
                    pass
                
                # *opt5_3* updates
                
                # if opt5_3 is starting this frame...
                if opt5_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    opt5_3.frameNStart = frameN  # exact frame index
                    opt5_3.tStart = t  # local t and not account for scr refresh
                    opt5_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(opt5_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'opt5_3.started')
                    # update status
                    opt5_3.status = STARTED
                    opt5_3.setAutoDraw(True)
                
                # if opt5_3 is active this frame...
                if opt5_3.status == STARTED:
                    # update params
                    pass
                
                # *opt5_4* updates
                
                # if opt5_4 is starting this frame...
                if opt5_4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    opt5_4.frameNStart = frameN  # exact frame index
                    opt5_4.tStart = t  # local t and not account for scr refresh
                    opt5_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(opt5_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'opt5_4.started')
                    # update status
                    opt5_4.status = STARTED
                    opt5_4.setAutoDraw(True)
                
                # if opt5_4 is active this frame...
                if opt5_4.status == STARTED:
                    # update params
                    pass
                
                # *rec_key1* updates
                waitOnFlip = False
                
                # if rec_key1 is starting this frame...
                if rec_key1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    rec_key1.frameNStart = frameN  # exact frame index
                    rec_key1.tStart = t  # local t and not account for scr refresh
                    rec_key1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(rec_key1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rec_key1.started')
                    # update status
                    rec_key1.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(rec_key1.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(rec_key1.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if rec_key1.status == STARTED and not waitOnFlip:
                    theseKeys = rec_key1.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                    _rec_key1_allKeys.extend(theseKeys)
                    if len(_rec_key1_allKeys):
                        rec_key1.keys = _rec_key1_allKeys[0].name  # just the first key pressed
                        rec_key1.rt = _rec_key1_allKeys[0].rt
                        rec_key1.duration = _rec_key1_allKeys[0].duration
                        # was this correct?
                        if (rec_key1.keys == str(str(int(correct_key)))) or (rec_key1.keys == str(int(correct_key))):
                            rec_key1.corr = 1
                        else:
                            rec_key1.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # *var_5* updates
                
                # if var_5 is starting this frame...
                if var_5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    var_5.frameNStart = frameN  # exact frame index
                    var_5.tStart = t  # local t and not account for scr refresh
                    var_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(var_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'var_5.started')
                    # update status
                    var_5.status = STARTED
                    var_5.setAutoDraw(True)
                
                # if var_5 is active this frame...
                if var_5.status == STARTED:
                    # update params
                    pass
                
                # *var* updates
                
                # if var is starting this frame...
                if var.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    var.frameNStart = frameN  # exact frame index
                    var.tStart = t  # local t and not account for scr refresh
                    var.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(var, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'var.started')
                    # update status
                    var.status = STARTED
                    var.setAutoDraw(True)
                
                # if var is active this frame...
                if var.status == STARTED:
                    # update params
                    pass
                
                # *var_6* updates
                
                # if var_6 is starting this frame...
                if var_6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    var_6.frameNStart = frameN  # exact frame index
                    var_6.tStart = t  # local t and not account for scr refresh
                    var_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(var_6, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'var_6.started')
                    # update status
                    var_6.status = STARTED
                    var_6.setAutoDraw(True)
                
                # if var_6 is active this frame...
                if var_6.status == STARTED:
                    # update params
                    pass
                
                # *var_7* updates
                
                # if var_7 is starting this frame...
                if var_7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    var_7.frameNStart = frameN  # exact frame index
                    var_7.tStart = t  # local t and not account for scr refresh
                    var_7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(var_7, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'var_7.started')
                    # update status
                    var_7.status = STARTED
                    var_7.setAutoDraw(True)
                
                # if var_7 is active this frame...
                if var_7.status == STARTED:
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
            if rec_key1.keys in ['', [], None]:  # No response was made
                rec_key1.keys = None
                # was no response the correct answer?!
                if str(str(int(correct_key))).lower() == 'none':
                   rec_key1.corr = 1;  # correct non-response
                else:
                   rec_key1.corr = 0;  # failed to respond (incorrectly)
            # store data for recognition_trials (TrialHandler)
            recognition_trials.addData('rec_key1.keys',rec_key1.keys)
            recognition_trials.addData('rec_key1.corr', rec_key1.corr)
            if rec_key1.keys != None:  # we had a response
                recognition_trials.addData('rec_key1.rt', rec_key1.rt)
                recognition_trials.addData('rec_key1.duration', rec_key1.duration)
            # the Routine "recognition_trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisRecognition_trial as finished
            if hasattr(thisRecognition_trial, 'status'):
                thisRecognition_trial.status = FINISHED
            # if awaiting a pause, pause now
            if recognition_trials.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                recognition_trials.status = STARTED
            thisExp.nextEntry()
            
        # completed 1 repeats of 'recognition_trials'
        recognition_trials.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "show_end_block" ---
        # create an object to store info about Routine show_end_block
        show_end_block = data.Routine(
            name='show_end_block',
            components=[end_block_text, end_block],
        )
        show_end_block.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        end_block_text.setText("Block " + str(session) + "/ 4 has ended \n\n You can have a break now! \n\n To continue, please press SPACE.")
        # create starting attributes for end_block
        end_block.keys = []
        end_block.rt = []
        _end_block_allKeys = []
        # store start times for show_end_block
        show_end_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        show_end_block.tStart = globalClock.getTime(format='float')
        show_end_block.status = STARTED
        thisExp.addData('show_end_block.started', show_end_block.tStart)
        show_end_block.maxDuration = None
        # keep track of which components have finished
        show_end_blockComponents = show_end_block.components
        for thisComponent in show_end_block.components:
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
        
        # --- Run Routine "show_end_block" ---
        thisExp.currentRoutine = show_end_block
        show_end_block.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock, 'status') and thisBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *end_block_text* updates
            
            # if end_block_text is starting this frame...
            if end_block_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                end_block_text.frameNStart = frameN  # exact frame index
                end_block_text.tStart = t  # local t and not account for scr refresh
                end_block_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_block_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_block_text.started')
                # update status
                end_block_text.status = STARTED
                end_block_text.setAutoDraw(True)
            
            # if end_block_text is active this frame...
            if end_block_text.status == STARTED:
                # update params
                pass
            
            # *end_block* updates
            waitOnFlip = False
            
            # if end_block is starting this frame...
            if end_block.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                end_block.frameNStart = frameN  # exact frame index
                end_block.tStart = t  # local t and not account for scr refresh
                end_block.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_block, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_block.started')
                # update status
                end_block.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(end_block.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(end_block.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if end_block.status == STARTED and not waitOnFlip:
                theseKeys = end_block.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _end_block_allKeys.extend(theseKeys)
                if len(_end_block_allKeys):
                    end_block.keys = _end_block_allKeys[-1].name  # just the last key pressed
                    end_block.rt = _end_block_allKeys[-1].rt
                    end_block.duration = _end_block_allKeys[-1].duration
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
                    currentRoutine=show_end_block,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                show_end_block.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if show_end_block.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in show_end_block.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "show_end_block" ---
        for thisComponent in show_end_block.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for show_end_block
        show_end_block.tStop = globalClock.getTime(format='float')
        show_end_block.tStopRefresh = tThisFlipGlobal
        thisExp.addData('show_end_block.stopped', show_end_block.tStop)
        # check responses
        if end_block.keys in ['', [], None]:  # No response was made
            end_block.keys = None
        blocks.addData('end_block.keys',end_block.keys)
        if end_block.keys != None:  # we had a response
            blocks.addData('end_block.rt', end_block.rt)
            blocks.addData('end_block.duration', end_block.duration)
        # the Routine "show_end_block" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisBlock as finished
        if hasattr(thisBlock, 'status'):
            thisBlock.status = FINISHED
        # if awaiting a pause, pause now
        if blocks.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            blocks.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'blocks'
    blocks.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "goodbye" ---
    # create an object to store info about Routine goodbye
    goodbye = data.Routine(
        name='goodbye',
        components=[goodbye_text, exit_exp],
    )
    goodbye.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for exit_exp
    exit_exp.keys = []
    exit_exp.rt = []
    _exit_exp_allKeys = []
    # store start times for goodbye
    goodbye.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    goodbye.tStart = globalClock.getTime(format='float')
    goodbye.status = STARTED
    thisExp.addData('goodbye.started', goodbye.tStart)
    goodbye.maxDuration = None
    # keep track of which components have finished
    goodbyeComponents = goodbye.components
    for thisComponent in goodbye.components:
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
    
    # --- Run Routine "goodbye" ---
    thisExp.currentRoutine = goodbye
    goodbye.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *goodbye_text* updates
        
        # if goodbye_text is starting this frame...
        if goodbye_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            goodbye_text.frameNStart = frameN  # exact frame index
            goodbye_text.tStart = t  # local t and not account for scr refresh
            goodbye_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(goodbye_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'goodbye_text.started')
            # update status
            goodbye_text.status = STARTED
            goodbye_text.setAutoDraw(True)
        
        # if goodbye_text is active this frame...
        if goodbye_text.status == STARTED:
            # update params
            pass
        
        # *exit_exp* updates
        waitOnFlip = False
        
        # if exit_exp is starting this frame...
        if exit_exp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            exit_exp.frameNStart = frameN  # exact frame index
            exit_exp.tStart = t  # local t and not account for scr refresh
            exit_exp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(exit_exp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'exit_exp.started')
            # update status
            exit_exp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(exit_exp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(exit_exp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if exit_exp.status == STARTED and not waitOnFlip:
            theseKeys = exit_exp.getKeys(keyList=['space', 'esc'], ignoreKeys=["escape"], waitRelease=False)
            _exit_exp_allKeys.extend(theseKeys)
            if len(_exit_exp_allKeys):
                exit_exp.keys = _exit_exp_allKeys[-1].name  # just the last key pressed
                exit_exp.rt = _exit_exp_allKeys[-1].rt
                exit_exp.duration = _exit_exp_allKeys[-1].duration
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
                currentRoutine=goodbye,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            goodbye.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if goodbye.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in goodbye.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "goodbye" ---
    for thisComponent in goodbye.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for goodbye
    goodbye.tStop = globalClock.getTime(format='float')
    goodbye.tStopRefresh = tThisFlipGlobal
    thisExp.addData('goodbye.stopped', goodbye.tStop)
    # check responses
    if exit_exp.keys in ['', [], None]:  # No response was made
        exit_exp.keys = None
    thisExp.addData('exit_exp.keys',exit_exp.keys)
    if exit_exp.keys != None:  # we had a response
        thisExp.addData('exit_exp.rt', exit_exp.rt)
        thisExp.addData('exit_exp.duration', exit_exp.duration)
    thisExp.nextEntry()
    # the Routine "goodbye" was not non-slip safe, so reset the non-slip timer
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
