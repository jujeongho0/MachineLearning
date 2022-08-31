import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, AveragePooling2D, ConvLSTM2D, Flatten, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
from base64 import b64encode
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad

def read_txt(fileName):
    with open(fileName, 'rt') as f:
        list_data = [a.strip('\n\r') for a in f.readlines()]
    return list_data

def write_json(fileName, data):
    with open(fileName, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_key(key_path):
    with open(key_path, "rb") as f:
        key = f.read()
    return key

def encrypt_data(key_path, ans_list, encrypt_store_path='ans.json'):
    key = load_key(key_path)
    data = " ".join([str(i) for i in ans_list])
    encode_data = data.encode()
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(encode_data, AES.block_size))
    iv = b64encode(cipher.iv).decode('utf-8')
    ct = b64encode(ct_bytes).decode('utf-8')
    write_json(encrypt_store_path, {'iv':iv, 'ciphertext':ct})

if __name__=="__main__":
    
    train_data_path = "train_data.npy" 
    train_label_path = "train_label.npy" 
    test_data_path = "test_data.npy"
    test_label_path = "test_label.npy" 

    x_train = np.load(train_data_path, allow_pickle=True) # (2682,) max = 502
    y_train = np.load(train_label_path, allow_pickle=True)  #(2682,)
    x_test = np.load(test_data_path, allow_pickle=True) # (2690,) max = 432
    y_test = np.load(test_label_path, allow_pickle=True) # (2690,)

    # 라벨 정수 인코딩
    label = []
    train_label = []
    test_label = []
    rating = ['18k', '17k', '16k', '15k', '14k', '13k', '12k', '11k', '10k', 
            '9k', '8k', '7k', '6k', '5k', '4k', '3k', '2k', '1k',
            '1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d']
    for i in range(2682):
        label.append(rating.index(y_train[i])) 

    for i in range(2690):
        label.append(rating.index(y_test[i]))

    label = np.array(label) 
    label= to_categorical(label)

    length = 50
    train_encoded = np.zeros((5372,length,19,19))

    for i in range(2682):
        for j in range(length):
            try:
                for row in range(0,19):
                    for col in range(0,19):
                        if x_train[i][j][0][row][col] == -1 :
                            train_encoded[i][j][row][col] = 200
                        elif x_train[i][j][0][row][col] == 1 :
                            train_encoded[i][j][row][col] = 100
                        else:
                            train_encoded[i][j][row][col] = x_train[i][j][0][row][col]
            except:
                continue

    for i in range(2690):
        for j in range(length):
            try:
                for row in range(0,19):
                    for col in range(0,19):
                        if x_test[i][j][0][row][col] == -1 :
                            train_encoded[i+2682][j][row][col] = 200
                        elif x_test[i][j][0][row][col] == 1 :
                            train_encoded[i+2682][j][row][col] = 100
                        else:
                            train_encoded[i+2682][j][row][col] = x_test[i][j][0][row][col]
            except:
                continue

    query_path = "query_data.npy" 
    query = np.load(query_path, allow_pickle=True) 

    query_list = np.zeros((2690,length,19,19))

    for i in range(2690):
        for j in range(length):
            try:
                for row in range(0,19):
                    for col in range(0,19):
                        if query[i][j][0][row][col] == -1 :
                            query_list[i][j][row][col] = 200
                        elif query[i][j][0][row][col] == 1 :
                            query_list[i][j][row][col] = 100
                        else:
                            query_list[i][j][row][col] = query[i][j][0][row][col]
            except:
                continue
                
    query_list = query_list.reshape((2690,length,19,19,1))

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')

    train_encoded = train_encoded.reshape(5372,length,19,19,1)
    x_train, x_test, y_train, y_test = train_test_split(train_encoded, label, test_size=.1, random_state=777)

    with tf.device('/device:GPU:0'):
        
        predict_list = []
        for i in range(10):
            model = Sequential()
            
            model.add(ConvLSTM2D(filters = 64, kernel_size = (4,4), return_sequences = True, input_shape = (length, 19, 19, 1)))  
            model.add(Activation('relu'))    
            model.add(TimeDistributed(AveragePooling2D(2,2)))
            
            model.add(ConvLSTM2D(filters = 64, kernel_size = (4, 4), return_sequences = True))   
            model.add(Activation('relu'))
            model.add(TimeDistributed(AveragePooling2D(5,5)))   

            model.add(Flatten())
            
            model.add(Dense(960))
            model.add(Activation('relu'))

            model.add(Dense(480))
            model.add(Activation('relu'))

            model.add(Dense(240))
            model.add(Activation('relu'))
            
            model.add(Dense(27, activation = "softmax"))
            
            model.summary()
            
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=12)
            mc = ModelCheckpoint('ML_team1.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
            model.compile(loss='categorical_crossentropy', optimizer= 'RMSProp' , metrics=['acc'])

            history = model.fit(x_train, y_train, batch_size=16, callbacks=[es, mc], epochs=90, validation_data=(x_test, y_test))

            predict_list.append(model.predict(query_list))
            
    predict_result = (predict_list[0] + predict_list[1] + predict_list[2] +predict_list[3] +predict_list[4] +predict_list[5] +predict_list[6] +predict_list[7] +predict_list[8] + predict_list[9])/10
    predict_label = []
    for k in predict_result:        
        predict_label.append(k.argmax())

    pred_test_label_txt = list_data = [str(rating[int(a)]).strip('\n\r') for a in predict_label]
    print(pred_test_label_txt)

    print(predict_result)
    np.save("prob.npy",predict_result)
    
    key_path = "team1.pem"
    ans = pred_test_label_txt
    encrypt_ans_path = "jeongho_ml.json"
    encrypt_data(key_path, ans, encrypt_ans_path)