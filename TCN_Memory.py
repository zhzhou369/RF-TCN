import numpy as np
import pandas as pd
import pylab as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import time

batch_size, time_steps, input_dim = None,32,1
def get_x_y():

    # 读取数据
    input_data_x = pd.read_excel(r'./Memory（3120）.xlsx', sheet_name='Sheet1', header=None)
    input_data_y = pd.read_excel(r'./Memory_energy（3120）.xlsx', sheet_name='Sheet1', header=None)
    input_data_y = input_data_y[0].tolist()  # 转成list集合

    # 归一化
    sc = MinMaxScaler(feature_range=(0, 1))
    input_data_x = sc.fit_transform(input_data_x)  # fit_transform 训练+转换

    input_data_x = input_data_x.reshape(3120, 32, 1)
    train_x, test_x, train_y, test_y = train_test_split(input_data_x, input_data_y, test_size=0.1,  shuffle=False)
    train_y = np.array(train_y)

    return train_x, train_y,test_x,test_y

train_x, train_y,test_x,test_y= get_x_y()

tcn_layer = TCN(input_shape=(time_steps, input_dim))
print('Receptive field size =', tcn_layer.receptive_field)

m = Sequential([
    tcn_layer,
    Dense(1)
])

#tcn_full_summary(m, expand_residual_blocks=False)
#损失函数（均方根误差）
m.compile(optimizer='Adam',
              loss='mean_squared_error')

tcn_full_summary(m, expand_residual_blocks=False)
m.fit(train_x, train_y, epochs=150, validation_split=0.2)

# m.save('tcn_net.h5')
# t1 = time.time()
# m = load_model('tcn_net.h5', custom_objects={'TCN': TCN})

y_predict = m.predict(test_x)
y_predict = list(y_predict)

# t2 = time.time()
# t3 = t2 - t1
# print("time:")
# print(t3)

df_data1 = pd.DataFrame(y_predict)
outputfile1 = "Predict_Memory_energy.xlsx"
df_data1.to_excel(outputfile1)

df_data2 = pd.DataFrame(test_y)
outputfile2 = "True_Memory_energy.xlsx"
df_data2.to_excel(outputfile2)

plt.figure(1)
plt.plot(test_y, color='green', label='true')
plt.plot(y_predict, color='red', label='predict')
plt.xlabel('time')
plt.ylabel('nenergy')
plt.title('predict')
plt.show()

print("done  ")
print('均方误差：',mean_squared_error(test_y, y_predict))
print('平均绝对误差：',mean_absolute_error(test_y, y_predict))

