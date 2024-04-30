"""
Created on Tue Apr  6 13:03:06 2021

@author: rpp
"""

from scipy.fft import rfft
import librosa #https://librosa.org/    #sudo apt-get install -y ffmpeg (open mp3 files)
import librosa.display
import librosa.beat
import sounddevice as sd  #https://anaconda.org/conda-forge/python-sounddevice
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance, cityblock
from scipy.stats import skew, kurtosis,pearsonr
import os


# Calcular as 7 estatísticas típicas sobre as features anteriores: média, desvio
# padrão, assimetria (skewness), curtose (kurtosis), mediana, máximo e mínimo
# (para efeitos de debug, sugere-se a utilização desta ordem). Para o efeito, utilizar
# a biblioteca scipy.stats (e.g., scipy.stats.skew).
# - Guarde as features num array numpy 2D, com número de linhas = número de
# músicas e número de colunas = número de features

# 2.1.1
def extractFeatures(file):
    y, sr = librosa.load(file)
    
    # features espectrais
    mfcc = librosa.feature.mfcc(y=y,n_mfcc=13)
    mfcc = calculate_stats(mfcc,1).flatten()
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y)
    spectral_centroid = calculate_stats(spectral_centroid,1).flatten()
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)
    spectral_bandwidth = calculate_stats(spectral_bandwidth,1).flatten()
    
    spectral_contrast = librosa.feature.spectral_contrast(y=y)
    spectral_contrast = calculate_stats(spectral_contrast,1).flatten()
    
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_flatness = calculate_stats(spectral_flatness,1).flatten()
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y)
    spectral_rolloff = calculate_stats(spectral_rolloff,1).flatten()
    
    # features temporais
    f0 = librosa.yin(y=y, fmin=20, fmax=11025)[0]
    f0 = calculate_stats(f0).flatten()
    
    rms = librosa.feature.rms(y=y)
    rms = calculate_stats(rms,1).flatten()
    
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    zero_crossing_rate = calculate_stats(zero_crossing_rate,1).flatten()
    
    # outras features
    tempo = librosa.feature.rhythm.tempo(y=y)
    
    features = np.concatenate([mfcc, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness, spectral_rolloff, f0, rms, zero_crossing_rate, tempo])

    return features


# 2.1.2
def calculate_stats(features, axis = None):
    mean = np.mean(features, axis = axis)
    std = np.std(features, axis = axis)
    skewness = skew(features, axis = axis)
    kurt = kurtosis(features, axis = axis)
    median = np.median(features, axis = axis)
    max_val = np.max(features,axis = axis)
    min_val = np.min(features, axis = axis)

    return np.array([mean, std, skewness, kurt, median, max_val, min_val])

# 2.1.3
def normalize_features(features):
    min_val = np.min(features, axis = 0)
    max_val = np.max(features, axis = 0)
    normalized_features = (features - min_val) / (max_val - min_val)
    normalized_features = np.vstack((max_val, normalized_features))
    normalized_features = np.vstack((min_val, normalized_features))
    return normalized_features

def calculate_sc(signal):
    spectrum = np.abs(rfft(signal))
    normalized_spectrum = spectrum / np.sum(spectrum)
    normalized_freqs = np.linspace(0, 1, len(normalized_spectrum))
    sc= np.sum(normalized_freqs * normalized_spectrum)
    return sc

def calculate_sc_librosa(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    return cent

def pearson_correlation(x, y):
    return pearsonr(x, y)[0]

def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

def euclidean_similarity(x, y):
    return 1 - distance.euclidean(x, y)

def cosine_similarity(x, y):
    return 1 - distance.cosine(x, y)

def manhattan_similarity(x,y):
    return 1 - cityblock(x, y)

if __name__ == "__main__":
    all_features = []
    for file in os.listdir('./Queries'):
        if file.endswith('.mp3'):
            print(file)
            features = extractFeatures('./Queries/' + file)
            all_features.append(features)

    #2.1.3
    # normalized_features = normalize_features(all_features)
    #2.1.4.
    np.savetxt('./resultadosObtidos/features.csv', features, delimiter = ',')
    # np.savetxt('./resultadosObtidos/features_norm.csv', normalized_features, delimiter = ',')
    #np.savetxt('./resultadosObtidos/.csv', all_features, delimiter = ',')
    


    plt.show()
    