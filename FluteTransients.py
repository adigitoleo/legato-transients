import numpy as np
import TransferFunctions as tf
import scipy.signal as sig
import audacity
import peakutils

###-+-+-+-+-+
### Functions for TF resonance analysis
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
### Functions for static geometry frequency analysis
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
### Functions for live transient analysis
###-+-+-+-+ 

def IRcalib(A, coeff):
    """
    A must be a numpy array
    """
    (a,b,c) = coeff
    gradIR = np.gradient(A)
    keyx = (np.sqrt(b**2-4*a*(c-A))-b)/(2*a)*(2.5e-04)
    gradX = 1/(np.sqrt(b**2-4*a*(c-A)))
    keyv = gradX*gradIR
    
    return keyx,keyv

def Pcalib(A, coeff):
    """
    A must be a numpy array
    """
    (a,b) = coeff
    P = a*A+b
    return P

def get_vpeaks(v, t, thresh, minsep, pk='max'):
    """
    """
    vabs = np.abs(v)
    i_bad = [ i for i, tval in enumerate(t)
             if tval < 1. or tval > t[-1]-1. ]
    vabs[i_bad] = 0
    peaks = peakutils.indexes(vabs, thres=thresh, min_dist=minsep)
    
    if pk=='max':
        vpeaks = [ v[i] for i in peaks if v[i] > 0]
        tpeaks = [ t[i] for i in peaks if v[i] > 0]
    elif pk=='min':
        vpeaks = [ v[i] for i in peaks if v[i] < 0]
        tpeaks = [ t[i] for i in peaks if v[i] < 0]
    else:
        raise ValueError(
            "Invalid peak type. Expected 'max' or 'min'")
    return vpeaks,tpeaks

def get_fpeaks(au, t, sr, est, sweep, opt=False):
    """
    """
    i_est = [ int(round(tval*sr)) for tval in est ]
    nsweep = int(round(sweep*sr))
    sweeplims = ([ indx-nsweep for indx in i_est ],
                 [ indx+nsweep for indx in i_est ])
    
    gradpeaks = []
    timepeaks = []
    audio = []
    zerox = []
    freqs = []
    for this, (start, stop) in enumerate(zip(sweeplims[0], sweeplims[1])):
        this_audio = au[start:stop]
        this_time = t[start:stop]
        
        # au[start+i-1] is the prev. sample
        # automatically skips half-cycle crosses
        i_zerocross = [ i for i, sample in enumerate(this_audio)
                       if sample > 0 and au[start+i-1] < 0 ]
        single_periods = [ this_time[i_zerocross[i]]-this_time[indx]
                          for i, indx in enumerate(i_zerocross, 1)
                         if i != (len(i_zerocross)) ]
        cycle_freqs = [ 1/period for period in single_periods ]
        
        # lowpass filter freqs. to remove sampling artefacts
        b, a = sig.butter(4, 1/8, 'low')
        filtfreq = sig.filtfilt(b, a, cycle_freqs, padtype='even')
        
        gradf = np.abs(np.gradient(filtfreq))
        gradf_peakval = np.max(gradf)
        
        # no check for multiple peaks, gradf_timeval *could* have len > 1
        gradf_timeval = [ this_time[i] for i, df in enumerate(gradf)
                         if df==gradf_peakval ]
        audio.append(this_audio)
        zerox.append(i_zerocross)
        freqs.append(cycle_freqs)
        gradpeaks.append(gradf_peakval)
        timepeaks.append(gradf_timeval)
    
    if opt==True:
        return audio, zerox, freqs
    else:
        timepeaks = np.unique(timepeaks).tolist()
        return gradpeaks, timepeaks
    
def get_ppeaks(au, t, sr, est, sweep):
    """
    Requires RMSWind
    """
    i_est = [ int(round(tval*sr)) for tval in est ]
    nsweep = int(round(sweep*sr))
    sweeplims = ([ indx-nsweep for indx in i_est ],
                 [ indx+nsweep for indx in i_est ])
    
    gradpeaks = []
    timepeaks = []
    for this, (start, stop) in enumerate(zip(sweeplims[0], sweeplims[1])):
        this_audio = au[start:stop]
        this_time = t[start:stop]
        pfilt, ptime = RMSWind(this_audio, sr=sr, nwind=512, nhop=256)
        # do i need to unpack ptime here?
        
        pgrad = np.abs(np.gradient(pfilt))
        pgrad_peakval = np.max(pgrad)
        pgrad_timeval = [ this_time[i] for i, dp in enumerate(pgrad)
                         if dp==pgrad_peakval ]
        gradpeaks.append(pgrad_peakval)
        timepeaks.append(pgrad_timeval)
    return gradpeaks, timepeaks
    
###-+-+-+-+-+
### Other useful functions/wrappers
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

# Andre - windowed RMS calculation
def RMSWind(x, sr=1, nwind=1024, nhop=512, windfunc=np.blackman):
    '''
    Calculates the RMS amplitude amplitude of x, in frames of
    length nwind, and in steps of nhop. windfunc is used as
    windowing function.
    nwind should be at least 3 periods if the signal is periodic.
    '''

    nsam = len(x)
    ist = 0
    iend = ist+nwind

    t = []
    ret = []

    wind = windfunc(nwind)
    wsum2 = np.sum(wind**2)

    while (iend < nsam):
        thisx = x[ist:iend]
        xw = thisx*wind

        ret.append(np.sum(xw*xw/wsum2))
        t.append(float(ist+iend)/2.0/float(sr))

        ist = ist+nhop
        iend = ist+nwind

    return np.sqrt(np.array(ret)), np.array(t)