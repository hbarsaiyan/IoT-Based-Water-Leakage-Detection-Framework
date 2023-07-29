import os
import glob
import pandas as pd
import pickle
import librosa
import librosa.display
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

for filepath in sorted(glob.iglob('./recordings/*.wav')):
    signal , sr = librosa.load(filepath)
    print(os.path.basename(filepath))
    mfccs = librosa.feature.mfcc(signal, n_mfcc=13, sr=sr)
    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs)
    pca = PCA(n_components = 1)
    mfccs_pca = pca.fit_transform(mfccs_scaled)
    MFCCS = mfccs_pca.transpose()
    X = pd.DataFrame(MFCCS)
    if (os.path.basename(filepath).startswith("T")):
        X[len(X.columns)] = 1
    else:
        X[len(X.columns)] = 0
    X.to_csv('data.csv', header=False, mode='a', index=False)

DF = pd.read_csv("data.csv")

scaler = StandardScaler()
DF_scaled = scaler.fit_transform(DF)
DF_scaled

pca = PCA(0.95)
DF_pca = pca.fit_transform(DF_scaled)
DF_pca.shape
DF_pca
pca.explained_variance_ratio_
DF.shape

x = DF.drop(DF.columns[-1], axis=1)
y = DF[DF.columns[-1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
pickle.dump(model, open('model.pkl', 'wb'))