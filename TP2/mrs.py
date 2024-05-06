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
from scipy import fft
from scipy.spatial import distance
from scipy.stats import skew, kurtosis
import os

music_num = 900


# 2.1.1
def calculateSpectralFeatures(y, fs):
    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=13)

    sc = librosa.feature.spectral_centroid(y = y)  #default parameters: sr = 22050 Hz, mono, window length = frame length = 92.88 ms e hop length = 23.22 ms 
    sc = sc[0]
    #plotFeature("Spectral Centroid",sc);    

    spec_bw=librosa.feature.spectral_bandwidth(y=y, sr=fs)
    spec_bw = spec_bw[0]
    #plotFeature("Spectral Bandwidth",spec_bw)

    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=fs)
    #plotFeature("Spectral Contrast",spec_contrast)
    
    spec_flatness = librosa.feature.spectral_flatness(y=y)
    spec_flatness = spec_flatness[0]
    #plotFeature("Spectral Flatness",spec_flatness)

    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=fs)
    spec_rolloff = spec_rolloff[0]
    #plotFeature("Spectral Rolloff",spec_rolloff)

    return mfcc, sc, spec_bw, spec_contrast, spec_flatness, spec_rolloff

def calculateTemporalFeatures(y, fs):
    #F0
    f0 = librosa.yin(y, fmin=20, fmax=11025)
    for i in range(len(f0)):
        if f0[i] is None:
            f0[i] = 0
    #plotFeature("F0",f0)
    #rms
    rms = librosa.feature.rms(y=y)
    rms = rms.reshape(-1)
    #plotFeature("RMS",rms)

    #zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr = zcr.reshape(-1)
    #plotFeature("Zero Crossing Rate",zcr)
    return f0, rms, zcr

def calculateFeatures(file):
    y, fs = librosa.load(file)
    mfcc, sc, spec_bw, spec_contrast, spec_flatness, spec_rolloff = calculateSpectralFeatures(y, fs)
    f0, rms, zcr = calculateTemporalFeatures(y, fs)
    tempo = librosa.feature.rhythm.tempo(y=y, sr=fs)

    #calcular estatisticas
    mfcc_stats = np.zeros((13,7))
    for i in range(mfcc.shape[0]):
        mfcc_stats[i] = calculate_stats(mfcc[i])
    mfcc_stats = mfcc_stats.flatten()
    sc_stats = calculate_stats(sc)
    spec_bw_stats = calculate_stats(spec_bw)
    spec_contrast_stats = np.zeros((7,7))
    for i in range(spec_contrast.shape[0]):
        spec_contrast_stats[i] = calculate_stats(spec_contrast[i])
    spec_contrast_stats = spec_contrast_stats.flatten()
    spec_flatness_stats = calculate_stats(spec_flatness)
    spec_rolloff_stats = calculate_stats(spec_rolloff)
    f0_stats = calculate_stats(f0)
    rms_stats = calculate_stats(rms)
    zcr_stats = calculate_stats(zcr)
    
    return np.concatenate([mfcc_stats, sc_stats, spec_bw_stats, spec_contrast_stats, spec_flatness_stats, spec_rolloff_stats, f0_stats, rms_stats, zcr_stats, tempo])

# 2.1.2
def calculate_stats(features):
    mean = np.mean(features)
    std = np.std(features)
    skewness = skew(features)
    kurt = kurtosis(features)
    median = np.median(features)
    max_val = np.max(features)
    min_val = np.min(features)

    return np.array([mean, std, skewness, kurt, median, max_val, min_val])

# 2.1.3
def normalize_features(features):
    min_max = np.zeros((2, 190))
    normalized_features = np.zeros((music_num+2, 190))
    for i in range(190):
        min_max[0][i] = np.min(features[:, i])
        min_max[1][i] = np.max(features[:, i])
        if min_max[1][i] == min_max[0][i]:
            normalized_features[2:, i] = 0
        else:
            normalized_features[2:, i] = (features[:, i] - min_max[0][i]) / (min_max[1][i] - min_max[0][i])
    normalized_features[0] = min_max[0]
    normalized_features[1] = min_max[1]
    return normalized_features

def normalize_query(query, min_max):
    normalized_query = np.zeros(190)
    for i in range(190):
        if min_max[1][i] == min_max[0][i]:
            normalized_query[i] = 0
        else:
            normalized_query[i] = (query[i] - min_max[0][i]) / (min_max[1][i] - min_max[0][i])
    return normalized_query

def calculate_sc(signal):
    spectrum = np.abs(rfft(signal))
    normalized_spectrum = spectrum / np.sum(spectrum)
    normalized_freqs = np.linspace(0, 1, len(normalized_spectrum))
    sc= np.sum(normalized_freqs * normalized_spectrum)
    return sc

def features():
    all_feats = np.zeros((music_num, 190))
    count = 0
    for file_name in os.listdir("./music"):
        if os.path.isfile(os.path.join("./music", file_name)):
            file_path = os.path.join("./music", file_name)
            # 2.1.1
            features = calculateFeatures(file_path)
            # print(features)
            # TODO: Guardar estatisitcas de cada feature
            all_feats[count] = features
            count += 1
            
    # 2.1.3
    norm_feats = normalize_features(all_feats)
    # 2.1.4
    np.savetxt('./resultadosObtidos/features.csv', all_feats, delimiter=',', fmt="%.6f")
    np.savetxt('./resultadosObtidos/features_norm.csv', norm_feats, delimiter=',', fmt="%.6f")

    return all_feats, norm_feats
    
#3.1
def distances(norm_feats):
    euclidean = np.zeros(music_num)
    manhattan = np.zeros(music_num)
    cos = np.zeros(music_num)
    # calculate features from query
    for file_name in os.listdir("./Queries"):
        if os.path.isfile(os.path.join("./Queries", file_name)):
            file_path = os.path.join("./Queries", file_name)
            query_features = calculateFeatures(file_path)
            # normalize query features
            query_features = normalize_query(query_features, norm_feats[:2])
            # calculate distances
            for i in range(music_num):
                euclidean[i] = distance.euclidean(norm_feats[i+2], query_features)
                manhattan[i] = distance.cityblock(norm_feats[i+2], query_features)
                cos[i] = distance.cosine(norm_feats[i+2], query_features)
            # save distances
            np.savetxt('./resultadosObtidos/euclidean.csv', euclidean, delimiter=',', fmt="%.6f")
            np.savetxt('./resultadosObtidos/manhattan.csv', manhattan, delimiter=',', fmt="%.6f")
            np.savetxt('./resultadosObtidos/cosine.csv', cos, delimiter=',', fmt="%.6f")

            #get 10 best results, save the indexes
            euclidean_best = np.argsort(euclidean)[:11]
            manhattan_best = np.argsort(manhattan)[:11]
            cos_best = np.argsort(cos)[:11]


            #get the names of the musics
            euclidean_names = []
            manhattan_names = []
            cos_names = []
            for i in euclidean_best:
                euclidean_names.append(os.listdir("./music")[i][:-4])
            for i in manhattan_best:
                manhattan_names.append(os.listdir("./music")[i][:-4])
            for i in cos_best:
                cos_names.append(os.listdir("./music")[i][:-4])

            #change indexes to distances
            euclidean_best = euclidean[euclidean_best]
            manhattan_best = manhattan[manhattan_best]
            cos_best = cos[cos_best]

            return euclidean_names, euclidean_best, manhattan_names, manhattan_best, cos_names, cos_best


#2.2.1
#spectral centroid calculated manually
#use scipy.fft.rfft
def spectralCentroid(y, sr):
    N = len(y)
    yf = np.fft.rfft(y)
    yf = np.abs(yf)
    yf = yf / np.sum(yf)
    f = np.fft.rfftfreq(N, 1/sr)
    sc = np.sum(f * yf)
    return sc

def manualCentroid():
    error = np.zeros((music_num, 2))
    count = 0
    for file_name in os.listdir("./music"):
        if os.path.isfile(os.path.join("./music", file_name)):
            file_path = os.path.join("./music", file_name)   
            warnings.filterwarnings("ignore")
            sr = 22050
            y, fs = librosa.load(file_path, sr=sr, mono = True)
            n_fft = 2048
            hop_length = 512
            sc = np.zeros((len(y)-n_fft+1)//hop_length +1)
            counter=0
            for i in range(0, len(y) - n_fft + 1, hop_length):
                df=sr/n_fft
                #freq=0,10,20,30,...
                freqs = np.arange(0, sr/2 + df, df)
                #hann window
                yw = np.hanning(n_fft)
                #FFT
                yf = fft.rfft(y[i:i+n_fft]*yw)
                #Espectro de potência
                magnitudes = np.abs(yf)
                #SC
                if np.sum(magnitudes)==0:
                    sc[counter]=(0)
                else:
                    sc[counter]=(np.sum(magnitudes*freqs)/np.sum(magnitudes))
                counter+=1
            librosa_sc=librosa.feature.spectral_centroid(y = y)[0][2:len(sc)+2]
            error[count] = (np.corrcoef(librosa_sc,sc)[0][1], np.sqrt(np.mean((librosa_sc-sc)**2)))
            count += 1
    np.savetxt('./resultadosObtidos/spectroid.csv', error, delimiter=',', fmt="%.6f")

#4.1.1
def rankingMetadata():
    queryMetadata = open('./query_metadata.csv', 'r').readlines()[1:]
    querySplit = queryMetadata[0].split(',')
    allMetadata = open('./panda_dataset_taffc_metadata.csv', 'r').readlines()[1:]
    allMetadataSplit = []
    for line in allMetadata:
       allMetadataSplit.append(line.split(','))
    # print(allMetadata)
    rankings = np.zeros(music_num)
    for i in range(music_num):
        #artist
        rankings[i] += (querySplit[1] == allMetadataSplit[i][1])
        #mood
        moods = querySplit[9].replace("\"","").replace(";","").split(" ")
        for m in moods:
            rankings[i] += (m in allMetadataSplit[i][9].replace("\"","").replace(";","").split(" "))
        #genre
        genres = querySplit[11].replace("\"","").replace(";","").split(" ")
        for g in genres:
            rankings[i] += (g in allMetadataSplit[i][11].replace("\"","").replace(";","").split(" "))
        
    indexes = np.argsort(rankings)[::-1][:11]
    names = []
    for i in indexes:
        names.append(allMetadataSplit[i][0].strip("\""))
    score = rankings[indexes]
    return names, score

#4.1.2
def precision(ranking, distance):
    count = 0
    for i in range(10):
        if ranking[i+1] in distance:
            count += 1
    return count / 10 * 100

#4.1.3
def printRanking(euclidean_names, euclidean_best, manhattan_names, manhattan_best, cos_names, cos_best, ranking_names, ranking_scores):
    rankingStr = "Ranking Metadata\n\n"
    rankingStr += "Euclidean: -------\n"
    rankingStr += "[" + ", ".join(str(name) for name in euclidean_names) + "]\n"
    rankingStr += "[" +", ".join(str(score) for score in euclidean_best) + "]\n\n"
    rankingStr += "Manhattan: -------\n"
    rankingStr += "[" + ", ".join(str(name) for name in manhattan_names) + "]\n"
    rankingStr += "[" + ", ".join(str(score) for score in manhattan_best) + "]\n\n"
    rankingStr += "Cosine: -------\n"
    rankingStr += "[" + ", ".join(str(name) for name in cos_names) + "]\n"
    rankingStr += "[" + ", ".join(str(score) for score in cos_best) + "]\n\n"
    rankingStr += "Metadata: -------\n"
    rankingStr += "[" + ", ".join(str(name) for name in ranking_names) + "]\n"
    rankingStr += "[" + ", ".join(str(score) for score in ranking_scores) + "]\n\n"
    rankingStr += "Precision: -------\n"
    rankingStr += "Euclidean: " + str(precision(ranking_names, euclidean_names)) + "\n"
    rankingStr += "Manhattan: " + str(precision(ranking_names, manhattan_names)) + "\n"
    rankingStr += "Cosine: " + str(precision(ranking_names, cos_names)) + "\n"
    
    file = open('./resultadosObtidos/ranking.txt', 'w')
    file.write(rankingStr)


def featuresRead():
    all_feats = np.loadtxt('./resultadosObtidos/features.csv', delimiter=',')
    norm_feats = np.loadtxt('./resultadosObtidos/features_norm.csv', delimiter=',')
    music_num = all_feats.shape[0]
    return all_feats, norm_feats

if __name__ == "__main__":
    # ft = features()
    # features()
    all_feats, norm_feats = featuresRead()
    # manualCentroid()
    euclidean_names, euclidean_best, manhattan_names, manhattan_best, cos_names, cos_best = distances(norm_feats)
    ranking_names, ranking_scores = rankingMetadata()
    printRanking(euclidean_names, euclidean_best, manhattan_names, manhattan_best, cos_names, cos_best, ranking_names, ranking_scores)



    # plt.show()
    
    