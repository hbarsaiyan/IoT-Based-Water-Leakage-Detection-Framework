import os
import time
import pickle
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import telepot
token = "your_token" # telegram token
receiver_id = receiver_id # telegram receiver id
bot = telepot.Bot(token)

import warnings
warnings.filterwarnings('ignore')

start_time = time.time()
while True:
    # Record the audio sample
    os.system("arecord -d 5 -r 48000 -t wav test.wav")
    file = "test.wav"
    signal , sr = librosa.load(file)
    # MFCC
    mfccs = librosa.feature.mfcc(signal, n_mfcc=13, sr=sr)
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(mfccs)
    # PCA
    pca = PCA(n_components = 1)
    X_pca = pca.fit_transform(X_scaled)
    MFCCS = X_pca.transpose()
    # Saving model
    model = pickle.load(open('model.pkl', 'rb'))
    result = model.predict(MFCCS)
    print(result[0])
    # Condition for sending message
    if (result[0] == 0):
        bot.sendMessage(receiver_id, "WATER LEAKAGE DETECTED!")
    time.sleep(60.0 - ((time.time() - start_time) % 60.0))