import os
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout

#Для развёрнутого показа np массива
np.set_printoptions(suppress=True)

#подгружаем сеты на 3 ситуации
s1 = np.loadtxt("data\\s1.txt")
s2 = np.loadtxt("data\\s2.txt")
s3 = np.loadtxt("data\\s3.txt")

All = np.array([])

#загоняем всё в один
All = np.vstack([s1, s2])
All = np.vstack([All, s3])
np.random.shuffle(All)
#разбиваем данные c - это наши 20 свойств, f наши состояния
c = All[:,0:20]
f = All[:,20]
print(c[0],f[0])


#создаём модель
model = keras.Sequential()

#делаем 5 слоёв(первый слой 126 нейрона,второй 64,12,выходной 1 нейрон)Dropout для ум. ошибки функция активации везде сигмойда
model.add(Dense(units=126, activation='sigmoid'))
model.add(Dense(units=64, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(units=12, activation='sigmoid'))
model.add(Dense(units=1, activation='sigmoid'))



#компилим модель, задаём потери как квадрат потерь, оптимизатор используем адам с шагом 0.01
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.01))

#запускаем обучение на 2000 эпох
history = model.fit(c, f,batch_size = 20, epochs= 2000, verbose=0, validation_split = 0.2)
model.summary()
#batch_size после каждых 20 входных значений корректируем коэффициенты
#validation_split делит на собственно обучающую и проверочную(20% случайных значений из обучающей в валидацию(увидим показатель качества))

testS1 = np.loadtxt("data\\testS1.txt")

#тут вставляем то, что хотим проверить
print(model.predict([[1,0,0,1,1,1,0,0,0,0,1,1,0,1,0,1,0,1,0,0], #1 которое добавил
[1,0,1,1,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0,1], #1 которое  не добавил 
[1,1,0,0,1,0,1,0,0,1,0,1,1,1,1,0,0,0,1,0], #2 состояние , которое добавил
[1,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0],#2 состояние , которое добавил
[1,0,1,1,0,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0], #3 состояние , которое не добавил
[0,0,1,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,1,0],#3 состояние , которое не добавил
[0,1,1,0,1,1,0,1,1,0,1,0,1,1,1,0,1,0,0,0],#3 состояние , которое не добавил
[0,0,1,0,0,1,1,0,0,1,1,1,1,0,0,0,0,1,0,0]])) #3 состояние , которое не добавил

print(model.predict([testS1]))
#print(model.get_weights())

#для выовда графиков потерь
plt.plot(history.history['loss'])
plt.grid(True)
plt.show()

model.save('modelReady')