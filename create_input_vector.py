import librosa
import numpy as np
import pandas as pd
import scipy

# A function that consolidates all of the below feature extraction functions
def create_feature_input(audio):

    y, sr = librosa.load(audio, sr=16000, dtype=np.float32)  # read the wav files

    y_harmonic, _ = librosa.effects.hpss(y)

    #features of audio signal
    RMS = librosa.feature.rms(y=y)
    CENs = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    MFCCs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)
    ZCR = librosa.feature.zero_crossing_rate(y_harmonic)
    tempo, _ = librosa.beat.beat_track(y=y_harmonic, sr=sr)

    #create dataframes
    RMSDF = make_RMSDF(RMS)
    ChromaDF = make_CENsDF(CENs)
    SpectralDF = make_SpectralDF(spectral_centroid, spectral_contrast, spectral_rolloff, spectral_flatness, spectral_bandwidth)
    MFCCsDF = make_MFCCsDF(MFCCs)
    ZCRDF = make_ZCRDF(ZCR)
    BeatDF = pd.DataFrame([{'tempo': tempo}])

    vector_input = pd.concat((ChromaDF, MFCCsDF, RMSDF, SpectralDF, ZCRDF, BeatDF), axis=1)
    return vector_input

def make_ZCRDF(ZCR):

    ZCR_mean = np.mean(ZCR)
    ZCR_std = np.std(ZCR)
    ZCR_skew = scipy.stats.skew(ZCR, axis=1)[0]

    data = [{'zcr mean': ZCR_mean,
            'zcr std': ZCR_std,
            'zcr skew': ZCR_skew}]
    ZCRDF = pd.DataFrame(data)

    return ZCRDF

def make_RMSDF(RMS):

    RMS_range = np.ptp(RMS)
    RMS_std = np.std(RMS)
    RMS_skew = scipy.stats.skew(RMS, axis=1)[0]

    data = [{'rms range': RMS_range,
             'rms std': RMS_std,
             'rms skew': RMS_skew}]
    RMSDF = pd.DataFrame(data)

    return RMSDF

def make_CENsDF(CENs):

    CENs_means = np.mean(CENs, axis=1)
    CENs_stds = np.std(CENs, axis=1)

    CENsDF = pd.DataFrame()


    for i in range(0, 12):
        name = 'chroma' + str(i) + ' mean'
        CENsDF[name]= CENs_means[i]
        CENsDF.loc[0,name] = CENs_means[i]

    for i in range(0, 12):
        name = 'chroma' + str(i) + ' std'
        CENsDF[name] = CENs_stds[i]
        CENsDF.loc[0,name] = CENs_stds[i]

    return CENsDF

def make_MFCCsDF(MFCCs):

    MFCCs_means = np.mean(MFCCs, axis=1)
    MFCCs_stds = np.std(MFCCs, axis=1)

    MFCCsDF = pd.DataFrame()


    for i in range(0, 13):
        name = 'mfccs' + str(i) + ' mean'
        MFCCsDF[name] = MFCCs_means[i]
        MFCCsDF.loc[0, name] = MFCCs_means[i]

    for i in range(0, 13):
        name = 'mfccs' + str(i) + ' std'
        MFCCsDF[name] = MFCCs_stds[i]
        MFCCsDF.loc[0, name] = MFCCs_stds[i]

    return MFCCsDF

def make_SpectralDF(spectral_centroid, spectral_contrast, spectral_rolloff, spectral_flatness, spectral_bandwidth):

    # spectral centroids stats
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_std = np.std(spectral_centroid)
    spectral_centroid_skew = scipy.stats.skew(spectral_centroid, axis=1)[0]

    # spectral contrasts stats
    spectral_contrast_means = np.mean(spectral_contrast, axis=1)
    spectral_contrast_stds = np.std(spectral_contrast, axis=1)

    # spectral rolloff points stats
    spectral_rolloff_mean = np.mean(spectral_rolloff)
    spectral_rolloff_std = np.std(spectral_rolloff)
    spectral_rolloff_skew = scipy.stats.skew(spectral_rolloff, axis=1)[0]

    # spectral flatness stats
    spectral_flatness_mean = np.mean(spectral_flatness)
    spectral_flatness_std = np.std(spectral_flatness)
    spectral_flatness_skew = scipy.stats.skew(spectral_flatness, axis=1)[0]

    # bandwidth stats
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_std = np.std(spectral_bandwidth)
    spectral_bandwidth_skew = scipy.stats.skew(spectral_bandwidth, axis=1)[0]

    SpectralDF = pd.DataFrame()

    data = [{'spectral centroid mean':spectral_centroid_mean,
             'spectral centroid std':spectral_centroid_std,
             'spectral centroid skew':spectral_centroid_skew,
             'spectral flat mean':spectral_flatness_mean.astype('float64'),
             'spectral flat std':spectral_flatness_std.astype('float64'),
             'spectral flat skew':spectral_flatness_skew.astype('float64'),
             'spectral rolloff mean':spectral_rolloff_mean,
             'spectral rolloff std':spectral_rolloff_std,
             'spectral rolloff skew':spectral_rolloff_skew,
             'spectral bandwidth mean': spectral_bandwidth_mean,
             'spectral bandwidth std':spectral_bandwidth_std,
             'spectral bandwidth skew':spectral_bandwidth_skew}]

    SpectralDF = pd.DataFrame(data)

    for i in range(0, 7):
        name = 'contrast' + str(i) + ' mean'
        SpectralDF[name] = spectral_contrast_means[i]
        SpectralDF.loc[0, name] = spectral_contrast_means[i]

    for i in range(0, 7):
        name = 'contrast' + str(i) + ' std'
        SpectralDF[name] = spectral_contrast_stds[i]
        SpectralDF.loc[0, name] = spectral_contrast_stds[i]

    return SpectralDF

