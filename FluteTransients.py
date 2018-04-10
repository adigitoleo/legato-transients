import numpy as np
import TransferFunctions as tf
import scipy.signal as sig
import audacity

###-+-+-+-+-+
### Functions for TF analysis method
###-+-+-+-+ 

def get_tfdata(aupfile, *args, nfft, testing=False):
    """
    Utilizes the TransferFunctions package to:
    - pad the data and remove audio delays
    - estimate the transfer function from
      an external to an internal receiver
    
    Input
    -----
    aupfile : audacity project file
    nfft : window length for transfer function calculation
    testing : (optional) boolean, if True returns additional
              transfer functions from source to receivers
    *args: (optional) channel numbers starting with 0 
        in the following order...
        internal, external, source, infrared
        defaults are 0, 1, 2, 3
    
    Output (numpy array format)
    ------
    returns a dictionary of the data with the following keys:
    'tf' : transfer function amplitude data
    'f' : transfer function frequency data
    optional keys returned only if infrared channel is given:
    'ir' : infrared sensor RMS amplitude
    optional keys returned only if testing==True:
    'tf_src_int' : transfer function from source to internal
    'tf_src_ext' : transfer function from source to external
    'f_int', 'f_ext' : corresponding frequency values
    """
    auf = audacity.Aup(aupfile)
    print(aupfile)
    sr = auf.rate
    rawdata = []
    chnums = []
    maxlen = 0
    for chno in range(auf.nchannels):
        chnums.append(chno)
        rawdata.append(auf.get_channel_data(chno))
        maxlen = max(maxlen, len(rawdata[-1]))

    data = np.zeros((len(rawdata), maxlen))
    for chno, chdata in enumerate(rawdata):
        data[chno,:len(chdata)] = chdata
        
    chtags = []
    if len(args) == 0:
        if len(data) < 2 or len(data) > 4:
            raise ValueError("Unrecognised number of data channels.",
                             "Use *args to specify channel numbers")
        else:
            chtags = chnums
    
    else:
        chtags = list(args)
    
    chdict = {}
    chnames = ['int', 'ext', 'src', 'ir']
    for tag, num in zip(chtags, chnums):
        chdict[chnames[num]] = data[tag]
    
    outdict = {}
    tfxy,ff = tf.tfe(chdict['int'], chdict['ext'], Fs=sr, NFFT=nfft)
    outdict['tf'] = np.array(tfxy)
    outdict['f'] = np.array(ff)
    for chname, chdata in chdict.items():
        if chname=='ir':
            outdict['ir'] = np.sqrt(np.mean((chdata-np.mean(chdata))**2))
    
    if testing == True:
        delay = tf.determineDelay(
            chdict['src']/np.mean(
                chdict['src']), chdict['ext']/np.mean(chdict['ext']),
            maxdel=2**15)
        print("Delay: %d samples" %delay)
        chdict['src'] = np.roll(chdict['src'], delay)
        for chname, chdata in zip(chnames[0:2], [chdict['int'], chdict['ext']]):
            tfxy,ff = tf.tfe(chdata, chdict['src'], Fs=sr, NFFT=nfft)
            outdict['tf_src_%s' %chname] = np.array(tfxy)
            outdict['f_%s' %chname] = np.array(ff)
            
    return outdict

def filter_tfdata(tfdata, order, fcrit):
    """
    Utilizes the scipy.signal package to:
    - construct a butterworth filter
    - filter the transfer function data
      both forwards and backwards
      
    Change this function to use the desired filter.
    
    Input
    -----
    tfdata : transfer function amplitude data
    order : order of the Butterworth filter
    fcrit : critical frequency for the filter
    
    Output
    ------
    returns the filtered data, converted to sound level
    """
    b, a = sig.butter(order, fcrit)
    smoothtf = sig.filtfilt(b, a, tfdata, padtype='constant')
    tflevel = 20*np.log10(np.abs(smoothtf))
    return tflevel

def get_hf(tfdata, fvals, lflim, uflim):
    """
    """
    if len(tfdata) != len(fvals):
        raise ValueError("len(tfdata) and len(fvals) do not match")
        
    lindex = int(np.floor((len(fvals)/fvals[-1])*lflim))
    uindex = int(np.ceil((len(fvals)/fvals[-1])*uflim))
    tfslice = tfdata[lindex:uindex]
    fslice = fvals[lindex:uindex]
    
    hfy = np.amax(tfslice)
    hfx = fslice[np.argmax(tfslice)]
    return hfx, hfy

###-+-+-+-+-+
### Functions for F0 analysis method
###-+-+-+-+ 

def get_fsdata(aupfile, *args, nfft=1024):
    """
    Pads the data and utilizes the numpy.fft package to:
    - get the discrete fourier transforms of the (raw) audio
    - get the corresponding frequency axes
    
    Input
    -----
    aupfile : audacity project file
    nfft : number of samples to use for fft calculation
            (power of 2 recommended - defaults to 1024)
    *args: (optional) channel numbers starting with 0 
        in the following order...
        internal, external, labium, infrared
        defaults are 0, 1, 2, 3
    
    Output (numpy array format)
    ------
    returns a dictionary of the data with the following keys:
    'int_fs' : internal fourier spectrum data
    'ext_fs' : external fourier spectrum data
    'lab_fs' : labium fourier spectrum data
    'int_wf', 'ext_wf', 'lab_wf' : corresponding waveforms
    optional keys returned only if infrared channel is given:
    'ir' : infrared sensor RMS amplitude
    """
    auf = audacity.Aup(aupfile)
    print(aupfile)
    sr = auf.rate
    rawdata = []
    chnums = []
    maxlen = 0
    for chno in range(auf.nchannels):
        chnums.append(chno)
        rawdata.append(auf.get_channel_data(chno))
        maxlen = max(maxlen, len(rawdata[-1]))

    data = np.zeros((len(rawdata), maxlen))
    for chno, chdata in enumerate(rawdata):
        data[chno,:len(chdata)] = chdata
        
    chtags = []
    if len(args) == 0:
        if len(data) < 2 or len(data) > 4:
            raise ValueError("Invalid number of data channels.",
                             "Use *args to specify channel numbers")
        else:
            chtags = chnums
    
    else:
        chtags = list(args)
    
    chdict = {}
    chnames = ['int', 'ext', 'lab', 'ir']
    for tag, num in zip(chtags, chnums):
        chdict[chnames[num]] = data[tag]
            
    outdict = {}
    for chname, chdata in chdict.items():
        if chname=='ir':
            outdict['ir'] = np.sqrt(np.mean((chdata-np.mean(chdata))**2))
        else:
            fourier = np.fft.rfft(chdata, nfft)
            outdict[chname+'_fs'] = 20*np.log10(np.abs(fourier))
            outdict[chname+'_wf'] = chdata
    return outdict

def get_f0(wfdata, lflim, uflim, nfft=1024, sr=44100):
    """
    Uses the scipy.signal package to construct a flat top
    window function, which is multiplied with the waveform.
    Performs the real-input fft from the numpy.fft package, 
    and finds the frequency peak and its estimated power
    in a range restricted by lflim ad uflim (in Hz).
    
    Input
    -----
    wfdata : the waveform data array
    lflim, uflim : lower and upper frequency limits
    nfft : number of samples to use for windowing & fft
            (power of 2 recommended - defaults to 1024)
    
    Output
    ------
    f0x, f0y : frequency and power of the harmonic peak
                detected between lflim and uflim
    """
    window = sig.flattop(nfft)
    wfwind = [wf*wd for wf, wd in zip(wfdata, window)]
    spectrum = np.fft.rfft(wfwind, nfft)
    fvals = np.fft.rfftfreq(nfft, 1/sr)
    power = 20*np.log10(np.abs(spectrum))
    
    lindex = int(np.floor((len(fvals)/fvals[-1])*lflim))
    uindex = int(np.ceil((len(fvals)/fvals[-1])*uflim))    
    pslice = power[lindex:uindex]
    fslice = fvals[lindex:uindex]
    
    f0x = fslice[np.argmax(pslice)]
    f0y = np.amax(pslice)
    return f0x, f0y

###-+-+-+-+-+
### Other useful functions
###-+-+-+-+ 

def get_stdevs(data, meandata):
    """
    """
    acqdevs = []
    for acqno, acqdata in enumerate(data):
        acqdevs.append([])
        for pt, m in zip(acqdata, meandata):
            acqdevs[acqno].append((pt-m)**2)

    sumsqs = [sum(x) for x in zip(*acqdevs)]
    stdevs = [np.sqrt(x/(len(data)-1)) for x in sumsqs]
    return np.array(stdevs)