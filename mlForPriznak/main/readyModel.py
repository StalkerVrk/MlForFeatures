import os
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'

import numpy as np
import matplotlib.pyplot as plt 
from tensorflow import keras
from tensorflow.keras.layers import Dense

testS1 = np.loadtxt("data\\testCF.txt")

c_t = testS1[:,0:20]
f_t = testS1[:,20]

model_loaded = keras.models.load_model('modelReady')
model_loaded.evaluate(c_t, f_t)

testS1 = np.loadtxt("data\\testS1.txt")

end = model_loaded.predict([[1,0,1,1,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0,1]])
if end < 0.3:
    print("Это 1 случай")
elif end >=0.3 and end < 0.65:
    print("Это 2 случай")
else:
    print("Это 3 случай")
#print(model_loaded.predict([testS1]))
#print(model_loaded.predict([c_t]))

print(end)