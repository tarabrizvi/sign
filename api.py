#usr/bi/env/python3
from flask import Flask, render_template,request,jsonify
import base64
import logging
import io
import os
from PIL import Image
import numpy as np
np.random.seed(1337)  # for reproducibility
import os

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras import backend as K
#from SignatureDataGenerator import SignatureDataGenerator
import getpass as gp
import sys
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import random

import cv2

graph = tf.get_default_graph()
# Create a session for running Ops on the Graph.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

app = Flask(__name__)


#ecludian distance
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

#ecludiean dist shape

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes

    return (shape1[0], 1)

def contrastive_loss(y_true,y_pred):
    margin=1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def fe_extration(x):
    rawImage = cv2.imread(x)
    hsv = cv2.cvtColor(rawImage, cv2.COLOR_BGR2HSV)
    hue ,saturation ,value = cv2.split(hsv)
    retval, thresholded = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    medianFiltered = cv2.medianBlur(thresholded,5)
    gray_image = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(medianFiltered, (200, 200))
    return resized_image
def fe_extration1(x):
    image=x.resize((200, 200), Image.ANTIALIAS)
    image=np.array(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue ,saturation ,value = cv2.split(hsv)
    retval, thresholded = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    medianFiltered = cv2.medianBlur(thresholded,5)
    gray_image = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(medianFiltered, (200, 200))
    return resized_image

def create_base_network_signet(input_shape):
    seq = Sequential()
    seq.add(Convolution2D(96, 11, 11, activation='relu', name='conv1_1', subsample=(4, 4), input_shape= input_shape,
                        init='glorot_uniform', dim_ordering='tf'))
    seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(ZeroPadding2D((2, 2), dim_ordering='tf'))

    seq.add(Convolution2D(256, 5, 5, activation='relu', name='conv2_1', subsample=(1, 1), init='glorot_uniform',  dim_ordering='tf'))
    seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
    seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))

    seq.add(Convolution2D(384, 3, 3, activation='relu', name='conv3_1', subsample=(1, 1), init='glorot_uniform',  dim_ordering='tf'))
    seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))

    seq.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', subsample=(1, 1), init='glorot_uniform', dim_ordering='tf'))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
#    model.add(SpatialPyramidPooling([1, 2, 4]))
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(1024, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform'))
    seq.add(Dropout(0.5))

    seq.add(Dense(128, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform')) # softmax changed to relu

    return seq



img_height = 100
img_width = 100

input_shape=(img_height, img_width, 1)

# network definition
base_network = create_base_network_signet(input_shape)

input_a = Input(shape=(input_shape))
input_b = Input(shape=(input_shape))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)
#print(eucl_dist_output_shape.shape)
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)
# compile model
rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
adadelta = Adadelta()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])
fname = os.path.join('./log_dir/' , 'weights_sign'+'.hdf5')

from PIL import Image

model.load_weights(fname)


import sqlite3




def create_table():
    conn = sqlite3.connect('tutorial.db',check_same_thread=False)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS customer(id INTEGER , sign TEXT)")

def data_entry(s,y):
    conn = sqlite3.connect('tutorial.db',check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO customer(id,sign) VALUES(?,?)",(s,y))

    conn.commit()
    c.close()



def read_from_db(cid):
    conn = sqlite3.connect('tutorial.db',check_same_thread=False)
    c=conn.cursor()
    c.execute('SELECT sign FROM customer WHERE id ='+cid)
    data = c.fetchall()
    img=[]
    for row in data:
        #img2=Image.open("./static/upload/"+row[0])
        rawImage=image.load_img("./static/upload/"+row[0],grayscale = True,
                    target_size=(100,100))
        rawImage=image.img_to_array(rawImage)
        t=("./static/upload/"+row[0])
        return rawImage,t


def preprocess(img1,img2):
    #rawImage = image.load_img(img1,grayscale = True,target_size=(100,100))
    #rawImage=image.img_to_array(rawImage)

    #forged image
    image=img2.resize((100, 100), Image.ANTIALIAS)
    image=image.convert('L')
    image=np.array(image)
    image=np.reshape(image,(100,100,1))


    image_pairs=[]
    image_pairs += [[img1, image]]
    images = [np.array(image_pairs)[:,0], np.array(image_pairs)[:,1]]
    return images

create_table()
print(model.summary())
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check')
def check():
    return render_template('acc.html')


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        inputs =[]
        files = request.files.getlist('file[]')
        for file_obj in files:
            # Check if no files uploaded
            if file_obj.filename == '':
                if len(files) == 1:
                    return render_template('select_files.html')
                continue

        text = request.form['ID']

        entry = {}
        entry.update({'filename': file_obj.filename})
        try:
            img_bytes = io.BytesIO(file_obj.stream.getvalue())
            entry.update({'data':Image.open(img_bytes)})
            #entry.update({'data':Image.open(img_bytes)})
        except AttributeError:
            img_bytes=io.BytesIO(file_obj.stream.read())
            mg_b64=base64.b64encode(img_bytes.getvalue())#. encoding('ascii')
            #img_bytes = io.BytesIO(file_obj.stream.getvalue())
            entry.update({'data1':Image.open(img_bytes)})
            img_b64=base64.b64encode(img_bytes.getvalue()).decode()
            entry.update({"db_data":mg_b64})
            entry.update({'data':img_bytes})
            inputs.append(entry)

        for input_ in inputs:
            mg_b64=input_['filename']
            up_img=input_['data1']
            up_img.save("./static/upload/"+mg_b64)
            #print(x)

            #mg_b64=str(mg_b64)
            print(type(mg_b64))

            print(type(text))
            data_entry(int(text),mg_b64)
            # data_entry('1','2')
            #conn.close()
        #return jsonify({'ID': "id found",
        #                'image':"image uploded sucessfully"})
        return render_template('index.html')


@app.route("/predict", methods=['GET','POST'])
def prediction():
    #here for predition
    '''
    take customer id as input
    take respective Sign
    convert sign to pillow image format
    mg_b64=base64.b64encode(img_bytes.getvalue()).decode_prob
    take scanned image from web page
    put it into model then give predicted outputs
    original image ,scanned image,processed image of both,
    plot graph of svg
    '''
    #global conn
    #conn = sqlite3.connect('tutorial.db',check_same_thread=False)

    if request.method == 'POST':
        inputs =[]
        files = request.files.getlist('file[]')
        text = request.form['ID']
        for file_obj in files:
            # Check if no files uploaded
            if file_obj.filename == '':
                if len(files) == 1:
                    return render_template('select_files.html')
                continue
            entry ={}
            entry.update({'filename':file_obj.filename})
            print(entry)
            try:
                img_bytes =io.BytesIO(file_obj.stream.read())
                entry.update({'data':Image.open(img_bytes)})
            except AttributeError:
                img_bytes =io.BytesIO(file_obj.stream.read())
                entry.update({'data':Image.open(img_bytes)})
            img_b64=base64.b64encode(img_bytes.getvalue()).decode()
            entry.update({'img':img_b64})
            inputs.append(entry)
        outputs=[]
        #preprocess scanned image
        for input_ in inputs:
            x= input_['data']
            #fx=fe_extration1(x)#
            print("hello")
            img,p= read_from_db(str(text))
            #fy=fe_extration(p)
            #print(img)
            with open(p, "rb") as imageFile:
                str01 = base64.b64encode(imageFile.read()).decode()
            #with open(fy, "rb") as imageFile:
            #    fy1 = base64.b64encode(imageFile.read()).decode()

            X=preprocess(img,x)

            #X1=np.array(X1)
            with graph.as_default():
                out=model.predict(X)
                out=out[0][0]
                out=(out*100)
                outputs.append(out)
                print(out)


        return render_template('results.html',len=len(outputs),scan_img=input_["img"],org_img=str01,acc=outputs,id=text)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port,debug=True)
