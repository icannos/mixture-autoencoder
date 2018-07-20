
# coding: utf-8

# In[1]:


# Paramètre global

seq_length = 50


# In[2]:


# Pour charger les données matlab
import h5py
from scipy.io import loadmat


# In[3]:


# Les outils Keras
from keras.models import Sequential
from keras.layers import Dense


# In[4]:


# Chargement des données
d= {}
mat = loadmat("raw_data/celia_eeg.mat")

# On affiche ce qu'on a
print(list(mat.keys()))


# In[5]:


import keras

# On construit le vecteur contenant les cibles
Y = [t[0] for t in mat["OUTPUT_T"]]

print(Y)

# On construit un vecteur avec les dates d'apparition de la cible
target_date = [int(t[0]) for t in mat["LAT_T"]]

print(target_date)

# Construit les vecteurs binaires à partir d'un vecteur de catégories (entier)


# In[6]:


# Les données: 565 cas, avec 2788 pas de temps et les 48 électrodes
data = mat["DATA1"]


# In[7]:


X = []

print(data.shape)

(features, duration, experiences) = data.shape

# Petit test de syntaxe pour vérifier 
seq = data[:,-seq_length-target_date[0]:-target_date[0],0]

print(seq.shape)

print(seq.transpose().shape)


# In[8]:


# Construit les données d'entrées en prenant 150 pas de temps avant la cible

from numpy.random import normal

sigma = 100

# C'est long, on sauvegardera peut-être le résultat

n_Y = []

for i in range(experiences):  
    seq = data[:,-seq_length-target_date[i]:-target_date[i],i].transpose()
    n_Y.append(Y[i])
    X.append(seq)
    
    # On multiplie par 10 le jeu de données
    #for j in range(10):
    #    X.append(normal(seq, sigma))        
    #    n_Y.append(Y[i])
        
Y = n_Y 
n_Y = []
    

# On va sauvegarder le résultat pour pas avoir à le recalculer

import pickle

pickle.dump(X, open("X_celia.dat", "wb"))


# In[9]:



import pickle
import numpy as np

# Pour charger X (On en a pas besoin a priori vu qu'on a fait le gros calcul mais si on devait relancer ce notebook
# et qu'on ne voulait pas refaire le calcul, on pourrait directement exécuter le code ici pour avoir X)
X = pickle.load(open("X_celia.dat", "rb"))

# On va filtrer les signaux avec un passe bande 2Hz-60Hz (On copie le papier de Fedjaev)

# Les 2 "constructeurs" de filtres que l'on va utiliser
from scipy.signal import iirnotch, butter

# Cette fonction applique un filtre à une série temporelle "dans un sens et dans l'autre
# D'après ce que j'ai compris ça évite le déphasage
from scipy.signal import filtfilt

# Pour construire un filtre qui supprime une fréquence (retourne les)
def notch_filter(freq_to_remove, sampling_freq, Q):
    w0 = freq_to_remove / (sampling_freq/2) # Pulsation
    b, a = iirnotch(w0, Q)
    return b, a

# Pour construire un passe bande 
def bandpass_filter(low_freq, high_freq, sampling_freq, order):
    w0b = low_freq / (sampling_freq/2) 
    w0h = high_freq / (sampling_freq/2) 
    b, a = butter(order, [w0b, w0h], 'bandpass')
    return b,a

def bandstop_filter(low_freq, high_freq, sampling_freq, order):
    w0b = low_freq / (sampling_freq/2) 
    w0h = high_freq / (sampling_freq/2) 
    b, a = butter(order, [w0b, w0h], 'bandstop')
    return b,a

def highpass_filter(low_freq, sampling_freq, order):
    w0b = low_freq / (sampling_freq/2) 
    b, a = butter(order, w0b, 'highpass')
    return b,a

# Pour appliquer le filtre 
def apply_filter(b, a, xn):
    return filtfilt(b,a, xn, axis=1)

b,a = highpass_filter(1, 1024, 5)

#X = apply_filter(b, a, X)

b,a = notch_filter(50, 1024, 30)

#X = apply_filter(b, a, X)

b,a = bandpass_filter(70, 120, 1024, 5)

#X = apply_filter(b, a, X)

#b,a = bandstop_filter(150, 250, 1000, 5)

#X = apply_filter(b, a, X)


# On va applatir les données
n_X = []

X = np.asarray(X)
m = np.max(X)

for i in range(experiences):
    seq = X[i].reshape(seq_length*features)
    m = np.max(seq)
    n_X.append(seq / m)
    
X = n_X
n_X=[]


# In[10]:


import tensorflow as tf
import numpy as np


class mixture_autoencoder():
    def __init__(self, learning_rate=0.001, training_steps=3000, batch_size=256,
                 autoencoders_topology=(128,64,8), classifier_topology=(8, 8), input_dim=seq_length*48, 
                 num_clusters=8,
                autoencoders_activation=(tf.nn.tanh, tf.nn.tanh, tf.nn.tanh)):
        '''

        :param learning_rate:
        :param training_steps:
        :param batch_size:
        :param autoencoders_topology:
        :param input_dim:
        :param num_clusters:
        '''

        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.autoencoders_topology = autoencoders_topology
        self.classifier_topology = classifier_topology
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.autoencoder_activation = autoencoders_activation
        
        self.sample_entropy = tf.placeholder("float")
        self.batch_entropy = tf.placeholder("float")

        # Le batch en cours de traitement

        self.X = tf.placeholder("float", [None, input_dim])

        self.Y_true = self.X

        # encoders[k] correspond à l'encoder du kieme autoencoder / pareil pour le decoder
        self.encoders = []
        self.decoders = []
        self.autoencoders_weights = []
        self.Y_preds = []
        self.classifier_weights = {}

        self.losses = []

    def compile(self):
        # Weights initialization

        # On construit les autoencoders
        for k in range(self.num_clusters):
            self.autoencoders_weights.append({"encoder-input": tf.Variable(tf.random_normal([self.input_dim, self.autoencoders_topology[0]]))})

            for i in range(len(self.autoencoders_topology) - 1):
                self.autoencoders_weights[k]["encoder-w" + str(i)] =                     tf.Variable(tf.random_normal([self.autoencoders_topology[i], self.autoencoders_topology[i + 1]]))

            for i in range(len(self.autoencoders_topology) - 1, 0, -1):
                self.autoencoders_weights[k]["decoder-w" + str(i)] =                     tf.Variable(tf.random_normal([self.autoencoders_topology[i], self.autoencoders_topology[i - 1]]))

            self.autoencoders_weights[k]["decoder-output"] =                 tf.Variable(tf.random_normal([self.autoencoders_topology[0], self.input_dim]))

            self.encoders.append(self.build_encoder(self.X,k))
            self.decoders.append(self.build_decoder(self.encoders[k], k))

            self.Y_preds.append(self.decoders[k])


        self.z = tf.concat(self.encoders, 1)

        self.classifier_weights["classifier-input"] =             tf.Variable(tf.random_normal([self.num_clusters*self.autoencoders_topology[-1], 
                                          self.classifier_topology[0]]))

        for i in range(len(self.classifier_topology) - 1):
            self.classifier_weights["classifier-w" + str(i)] =                 tf.Variable(tf.random_normal([self.classifier_topology[i], self.classifier_topology[i + 1]]))

        self.classifier_weights["classifier-output"] =             tf.Variable(tf.random_normal([self.classifier_topology[-1], self.num_clusters]))

        self.classifier = self.build_classifier()

        for k in range(self.num_clusters):
            self.losses.append(self.build_cluster_loss(k))

        self.loss = self.build_loss()

        self.train_network = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.init = tf.global_variables_initializer()

    def build_cluster_loss(self, k):
        loss = tf.reduce_sum(tf.pow(self.Y_true - self.Y_preds[k], 2, name="cluster_loss"), 1) * self.classifier[:, k]
        return loss

    def build_loss(self):
        self.losses = tf.stack(self.losses)
        
        self.element_wise_entropy = - tf.reduce_sum(self.classifier * tf.log(self.classifier), 1)
        
        self.p_mean = tf.reduce_mean(self.classifier,0)

        self.batch_wise_entropy = - tf.reduce_sum(self.p_mean * tf.log(self.p_mean), 0)

        loss = tf.reduce_mean(tf.reduce_sum(self.losses, 0) + self.sample_entropy * self.element_wise_entropy)                - self.batch_entropy * self.batch_wise_entropy

        return loss

    def build_classifier(self):
        layer = tf.nn.sigmoid((tf.matmul(self.z, self.classifier_weights["classifier-input"])))

        for i in range(len(self.classifier_topology) - 1):
            layer = tf.nn.sigmoid((tf.matmul(layer, self.classifier_weights["classifier-w" + str(i)])))

        layer = tf.nn.softmax((tf.matmul(layer, self.classifier_weights["classifier-output"])))

        return layer




    def build_encoder(self, X, k):
        '''

        :param X:
        :return:
        '''
        layer = self.autoencoder_activation[0]((tf.matmul(X, self.autoencoders_weights[k]["encoder-input"])))

        for i in range(len(self.autoencoders_topology) - 1):
            layer = self.autoencoder_activation[i]((tf.matmul(layer, self.autoencoders_weights[k]["encoder-w" + str(i)])))

        return layer

    def build_decoder(self, X, k):
        '''

        :param X:
        :return:
        '''
        layer = X

        for i in range(len(self.autoencoders_topology) - 1, 0, -1):
            layer = self.autoencoder_activation[i]((tf.matmul(layer, self.autoencoders_weights[k]["decoder-w" + str(i)])))

        layer = self.autoencoder_activation[0]((tf.matmul(layer, self.autoencoders_weights[k]["decoder-output"])))

        return layer


# In[19]:


X = np.asarray(X)
Y = np.asarray(Y)



#X = np.concatenate((X1, X2, X3), axis=0)

import plotly as ply
import plotly.graph_objs as go

shuffle = np.arange(X.shape[0])
np.random.shuffle(shuffle)


import tensorflow as tf

model = mixture_autoencoder(learning_rate=0.001, training_steps=100, batch_size=16,
                            autoencoders_topology=(64, 16, 8), classifier_topology=(32, 16, 8, 3), input_dim=63*seq_length,
                            num_clusters=4)

model.compile()

sess = tf.Session()
sess.run(model.init)


# In[21]:




loss = 100
for epoch in range(100):
    np.random.shuffle(shuffle)
    print("Epoch", epoch)

    # tf.summary.FileWriter("summaries/graph",sess.graph)

    for j in range(X.shape[0] // model.batch_size - 2):
        _, loss, batch_wise, p_mean = sess.run([model.train_network, model.loss, model.batch_wise_entropy, model.p_mean], feed_dict=
        {model.X: X[shuffle][j * model.batch_size:(j + 1) * model.batch_size],
         model.sample_entropy: 15,
         model.batch_entropy: 50
         })

        print("Loss:", loss, "Batch wise entropy:", batch_wise)
        print(p_mean)




# In[16]:


data = sess.run([model.classifier], feed_dict={model.X: X})

for i in range(len(data[0])):
    print(data[0][i])


# In[14]:


from keras.layers import Dense, Dropout, Activation,GaussianNoise
from keras.optimizers import SGD
from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical


model_classifier = Sequential()
model_classifier.add(Dense(8, activation='relu', input_dim=8))
model_classifier.add(Dense(5, activation = "relu"))
model_classifier.add(Dense(4, activation = "softmax"))

# On le construit avec comme erreur: categorical_crossentropy (c'est pour la catégorisation)
model_classifier.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy'])


# In[15]:


### from keras.utils import to_categorical

vector_target = to_categorical(Y%4)

print(vector_target.shape)

history = model_classifier.fit(np.asarray(data[0]), np.asarray(vector_target), epochs=150, verbose=1, 
                    batch_size=4, validation_split=0.2,shuffle=True)

