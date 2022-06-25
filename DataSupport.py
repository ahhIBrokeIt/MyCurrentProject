###################################################
#
#  Data support module for processing
#  raw sound data into usable training dataset
#
###################################################

import numpy as np
import random
import glob
import soundfile as sf
import scipy.signal
from AudioPreProcess import *


def ReadWavFile(filename, downSampleFactor=None):
    '''
    read wav file --> returns audio, metadata
          audio: datatype numpy.array with shape (samples, channels)
                 audio array has dtype32 values in range (-1,1)
       metadata: tuple (samplerate, channels, subtype)

       downSampleFactor: (optional) integer value to decimate the audio
    '''
    wavFile = sf.SoundFile(filename)
    channels = wavFile.channels
    subtype = wavFile.subtype
    wavFile.close()
    audio, samplerate = sf.read(filename, dtype='float32')

    if downSampleFactor != None:
        audio = scipy.signal.decimate(audio, downSampleFactor)

    return audio, (samplerate, channels, subtype)


def WriteWavFile(filename, audio, samplerate, stype):
    '''
    write wav file
          audio: datatype numpy.array with shape (samples, channels)
                 audio array has dtype32 values in range (-1,1)
    '''

    sf.write(filename, audio, samplerate, subtype=stype)


def BuildNoteDict(filename, skip):
    '''Read text file of note <-> frequencies and return table as dictioary'''
    tableFile = open(filename, 'r')
    table = tableFile.readlines()
    tableFile.close()

    freqTab = dict()

    for line in table[skip:]:
        note, freq, _ = line.split(',')
        if '/' in note:
            notes = note.split('/')
            for n in notes:
                freqTab[n] = float(freq)
        else:
            freqTab[note] = float(freq)

    return freqTab


def pullFreqs(LowNote, HighNote, noteDict):
    '''pull interval of frequencies from noteDict'''
    lowerFreq, upperFreq = noteDict[LowNote], noteDict[HighNote]

    freqs = []
    for val in noteDict.values():
        if val >= lowerFreq and val <= upperFreq:
            freqs.append(val)

    return freqs


def glob_wav(path):
    '''glob up all the wav files from <path> folder and all subfolders'''
    return glob.glob(path + '/**/*.wav', recursive=True)


def progress_bar(pComplete, len=30):
    '''output progress bar as string'''
    pos = int(pComplete * len / 100)
    bar = '=' * (pos - 1) + '>' + '.' * (len - pos - 1)

    return '[' + bar + ']'


def chunkify(audioIn, audioOut, win_size, scaleLst, sampleRate, randomize=False, ranNum=None):
    '''
       input a pair of syncronized audio tracks audioIn, audioOut
       return stacked numpy arrays of windowed segments of size <win_size>.

       notes: the input tracks must be the same shape.
              the remainder of the audio is discarded.

       If randomize is set to True, the starting window position is randomized
        and will reutrn either the default integer quotient of windows
        or a specified <ranNum> number of windows segments.
    '''
    assert(np.shape(audioIn) == np.shape(audioOut))
    num_wins = len(audioIn) // win_size
    if randomize == True and ranNum != None:
        num_wins = ranNum

    aud_win_inp, aud_freq_inp, aud_win_out = [], [], []

    prev_str = ''
    for w in range(num_wins):
        if randomize == False:
            pos = w * win_size
        else:
            pos = random.randint(len(audioIn) - win_size)

        # update completion of prep
        pcnt_comp = 100 * w / num_wins
        load_str = progress_bar(pcnt_comp) + ' ' + str(int(pcnt_comp)) + '% '
        if prev_str != load_str:
            print('\r' + load_str, end='')
            prev_str = load_str

        this_one = audioIn[pos: pos + win_size]
        aud_win_inp.append(this_one)

        this_one = scalogram_coeffs(this_one, scaleLst, sampleRate)
        aud_freq_inp.append(this_one)

        this_one = audioOut[pos: pos + win_size]
        aud_win_out.append(this_one)

    # remove status from terminal
    print('\r' + len(load_str) * ' ', end='\r')

    # stack arrays for final form
    fin_inp = np.stack(aud_win_inp, axis=0)
    fin_inpf = np.stack(aud_freq_inp, axis=0)
    fin_out = np.stack(aud_win_out, axis=0)

    return fin_inp, fin_inpf, fin_out


def prep_instrument_dataset(path, dsSize, win_size, scales, shuffle=False, RandNum=None):
    '''recursively collects all wav-files from <path>
       and generates numpy input/target audio datasets
       for single instrument autoencoder training.
    '''

    print('Instrument training:')
    print('Collecting audio files ... ')

    audio_data = []

    files = glob_wav(path)
    if dsSize != None:
        if len(files) < dsSize:
            dsSize = len(files)
        files = files[0:dsSize]

    ind, tot_files = 0, len(files)

    for file in files:

        # load wav file and prep.
        audio, meta = ReadWavFile(file)
        print('file ', ind+1, '/', tot_files, ' - ', file, sep='')

        audio_in, audio_scale, audio_out = chunkify(
            audio, audio, win_size, scales, sampleRate=meta[0], randomize=shuffle, ranNum=RandNum)
        audio_data.append([audio_in, audio_scale, audio_out])
        ind += 1

    # remove status from terminal
    # print('\r' + len(load_str) * ' ', end='\r')

    if shuffle == True:
        random.shuffle(audio_data)

    # finalize training set
    audio_datain = [val[0] for val in audio_data]
    audio_scales = [val[1] for val in audio_data]
    audio_dataout = [val[2] for val in audio_data]

    data_in = np.vstack(audio_datain)
    data_sc = np.vstack(audio_scales)
    data_out = np.vstack(audio_dataout)

    print('done!')

    print(len(data_in), 'samples in training set')
    return data_in, data_sc, data_out


def save_dataset(**kwargs):
    '''save dataset: save numpy dataset.
       Each part of the dataset is saved seperately using keyword arguments
            input : input data set
           output : output data set
       instrument : name/type of instrument
             path : location other than current folder
    '''
    ds_size, path = 0, ''
    if 'path' in kwargs.keys():
        path = kwargs['path']
        if path[-1] != '/':
            path += '/'

    # aquire dataset properties
    if 'input' in kwargs.keys():
        ds_shape = np.shape(kwargs['input'])
    elif 'output' in kwargs.keys():
        ds_shape = np.shape(kwargs['output'])

    inst_in, inst_out = '', ''
    if 'instIn' in kwargs.keys():
        inst_in = kwargs['instIn']
    if 'instOut' in kwargs.keys():
        inst_out = kwargs['instOut']

    ds_size = str(ds_shape[0])
    dsname = inst_in + '-' + inst_out
    dsname += '-(' + ds_size + ')'
    in_name = path + 'input-' + dsname + '.npy'
    in_scales = path + 'input-scale-' + dsname + '.npy'
    out_name = path + 'output-' + dsname + '.npy'

    for name, val in kwargs.items():
        if name == 'input':
            np.save(in_name, val, allow_pickle=False, fix_imports=False)
        if name == 'input2':
            np.save(in_scales, val, allow_pickle=False, fix_imports=False)
        if name == 'output':
            np.save(out_name, val, allow_pickle=False, fix_imports=False)


def load_dataset(path, type, ds_size, inst_in, inst_out=None):
    '''load dataset from the save dataset method.
       identify data files with
           type : input/scale/output
        ds_size : dataset size
        inst_in : input instrument
       inst_out : output instrument (leave as None is same as input instument)
    '''
    assert(type in ['input', 'scale', 'output'])
    if type == 'scale':
        type = 'input-' + type

    if path[-1] != '/':
        path += '/'

    if inst_out == None:
        inst_out = '-' + inst_in

    dsname = inst_in + '-' + inst_out + '-(' + str(ds_size) + ')'
    filename = path + type + '-' + dsname + '.npy'

    return np.load(filename)
