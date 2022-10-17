from secrets import choice
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image,ImageOps
from io import BytesIO
from itertools import cycle
import time
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

import tensorflow as tf
import tensorflow_hub as hub
import os
from keras.utils import image_utils
from keras import models, layers,preprocessing
from keras.layers import Conv2D, MaxPooling2D, Flatten , Dense, Activation,Dropout,BatchNormalization
from keras.utils.vis_utils import plot_model

st.set_page_config(layout="wide")
st.set_option ('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.ion()
st.title("Automated Image Labelling Portal")
st.text("Upload Images for classification")

fig = plt.figure()

def main():

    img=[]
    file_name= []
    corrected_labels=[]
    labels = []
    class_names =['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    uploaded_files = st.file_uploader('', accept_multiple_files=True)

    with st.form("my_form"): 
        for uploaded_file in uploaded_files: 
            image = Image.open(uploaded_file)
            new_image = image.resize((600, 400))
            # st.image(new_image,width=200)
            img.append(new_image)

            with st.spinner('Interpretating model....'):        
                    interpretation1 = interpret1(image)
                    time.sleep(1)
                    st.pyplot(interpretation1)        
            with st.spinner('Model working....'):
                    # plt.imshow(image)
                    plt.axis("off")
                    predictions = predict(image)
                    time.sleep(1)
                    st.success('Classified')
                    st.write(predictions)               
            choice = st.selectbox("select an option from dropdown to rectify label for above image",['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'],key = uploaded_file)
            corrected_labels.append(choice)
        submitted = st.form_submit_button("Submit & Retrain")
        if submitted:

            for i in range(0, len(corrected_labels)):
                for j in range(0,9):
                    if corrected_labels[i] == class_names[j]:
                        # f.write(str(corrected_labels))
                        labels.append(j)
        y_train_new = np.asarray(labels)

        #try:
        rsize  = np.asarray(image.resize((32,32))).astype(np.float32)
        rsize= np.expand_dims(rsize,axis =0) # Use PIL to resize # Use PIL to resize     
        for i in range(0,len(uploaded_files)):
            if i == 0:
                X_train_new = rsize
            else:
                X_train_new = np.append(X_train_new, rsize, axis=0)

        import keras
        Cifar10=keras.datasets.cifar10 # Loading the dataset
        (xtrain,ytrain),(X_test,y_test)= Cifar10.load_data()
        print(xtrain.shape)
        print(ytrain.shape)
        print(X_test.shape)
        print(y_test.shape)
        # @st.cache(allow_output_mutation=True)
        from sklearn.model_selection import train_test_split
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = 0)
        print(X_test.shape,X_val.shape,y_test.shape,y_val.shape )

        def normalize(x):
            x = x.astype('float32')
            x = x/255.0
            return x 

        X_train_new = normalize(X_train_new)
        X_test = normalize(X_test)
        X_val = normalize(X_val) 
        # print(y_train_new.shape, y_train_new[0]))
        from tensorflow.keras.utils import to_categorical
        y_train_new =to_categorical(y_train_new , 10)
        y_test = to_categorical(y_test , 10)
        y_val  = to_categorical(y_val , 10)
        print((X_train_new.shape, X_test.shape,X_val.shape))
        print("y_train_new Shape: %s and value: %s" % (y_train_new.shape, y_train_new[0]))
        print("y_test Shape: %s and value: %s" % (y_test.shape, y_test[0]))
        print("y_val Shape: %s and value: %s" % (y_val.shape, y_val[0]))

        with st.spinner('Model is getting retrained....'):     
            epoch = 20
            #retrain_model = tf.keras.models.load_model('retrained_X_test100_79_model.h5')
            #if not os.path.isfile('retrained_X_test100_79_model.h5'):
            retrained_model = urllib.request.urlretrieve('https://github.com/ankan-mazumdar/Active-Learning2/blob/main/retrained_X_test100_79_model.h5', 'retrained_X_test100_79_model.h5')
            print('retrain_model=====',retrain_model)
            #else:
                    #print('no model.h5 for retraining found') 
            retrain_model = tf.keras.models.load_model(retrained_model)        
            es_callbacks=[tf.keras.callbacks.EarlyStopping(patience=0, verbose=1)]
            opt = tf.keras.optimizers.Adam(1e-3)
            # compile the model
            retrain_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            # r = retrain_model.fit(datagen.flow(X_train_new , y_train_new , batch_size = 32), epochs = epoch , validation_data = (X_val , y_val), verbose = True, callbacks=[cp_callback])
            r = retrain_model.fit(X_train_new, y_train_new, epochs= epoch, batch_size=32, validation_data = (X_val , y_val), verbose = True,callbacks=[es_callbacks])
            loss, acc = retrain_model.evaluate(X_train_new, y_train_new, verbose=1)
            print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
            retrain_model.save('retrained_Streamlit_model.h5') 


        for uploaded_file in uploaded_files: 
            image = Image.open(uploaded_file)

            with st.spinner('Interpretating retrained model....'):        
                    interpretation2 = interpret2(image)
                    time.sleep(1)
                    st.pyplot(interpretation2)        
            with st.spinner('retrained model predicting....'):
                    # plt.imshow(image)
                    plt.axis("off")
                    predictions = predict_retrain(image)
                    time.sleep(1)
                    st.success('Reclassification completed')
                    st.write(predictions)   
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])            
        loss_b, acc_b= model.evaluate(X_train_new, y_train_new, verbose=1)
        print('base model, accuracy: {:5.2f}%'.format(100 * acc_b))
        if acc > acc_b:
            st.write("accuracy has improved after model retrain, replacing old model by new retrained model")     
            classifier_model= retrain_model  
        else:
            st.write("accuracy has not improved after model retrain,retaining the same model")     
            # classifier_model= model                
            #with open(os.path.join('files/downloaded_images/',uploaded_file.name),"wb") as f: 
            #    f.write(uploaded_file.getbuffer()) 
        #except:
        #    st.error("Please upload images")

import urllib.request            
classifier_model = 'my_model.h5'
if not os.path.isfile('my_model.h5'):
        classifier_model = urllib.request.urlretrieve('https://github.com/ankan-mazumdar/Active-Learning2/blob/main/my_model.h5?raw=true', 'my_model.h5')
# model = models.load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
model = tf.keras.models.load_model(classifier_model)

if not os.path.isfile('retrained_Streamlit_model.h5'):
        retrain_model = urllib.request.urlretrieve('https://github.com/ankan-mazumdar/Active-Learning2/blob/main/my_model.h5?raw=true', 'retrained_Streamlit_model.h5')
else:
        retrain_model = tf.keras.models.load_model('retrained_Streamlit_model.h5')     
def predict_retrain(image):
 
    retrain_model = tf.keras.models.load_model('retrained_Streamlit_model.h5')
    test_image = image.resize((32,32))
    test_image = image_utils.img_to_array(test_image).astype(np.float32)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names =['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    predictions = retrain_model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()

    results = {'Airplane': 0, 'Automobile': 1, 'Bird': 2, 'Cat': 3, 'Deer': 4, 'Dog': 5, 'Frog': 6, 'Horse': 7, 'Ship': 8, 'Truck': 9}
    result = f"Predicted as  {class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result

def predict(image):
    
    test_image = image.resize((32,32))
    test_image = image_utils.img_to_array(test_image).astype(np.float32)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names =['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {'Airplane': 0, 'Automobile': 1, 'Bird': 2, 'Cat': 3, 'Deer': 4, 'Dog': 5, 'Frog': 6, 'Horse': 7, 'Ship': 8, 'Truck': 9}
    result = f"Predicted as  {class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result

def explain(image):
    rsize = np.asarray(image.resize((32,32))) # Use PIL to resize
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(rsize.astype('double'), model.predict,  top_labels=3, hide_color=0, num_samples=1000)
    return explanation
def interpret1(image):
    
    exp= explain(image)
    temp_1, mask_1 = exp.get_image_and_mask(exp.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    temp_2, mask_2 = exp.get_image_and_mask(exp.top_labels[0], positive_only=False, num_features=10, hide_rest=False)

    ind =  exp.top_labels[0]
    #Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(exp.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(exp.segments) 
 
    fig, (ax1, ax2, ax3, ax4 ) = plt.subplots(1, 4,  figsize=(15,15))   
    ax1.imshow(image)
    ax1.set_title('Original image')
    ax2.imshow(mark_boundaries(temp_1, mask_1))
    ax2.set_title('Superpixel boundaries')
    ax3.imshow(mark_boundaries(temp_2, mask_2))
    ax3.set_title('Superpixel boundaries')
    ax4.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    ax4.set_title('Heat Map')
    plt.tight_layout()
    plt.show()

def explain2(image):
    rsize = np.asarray(image.resize((32,32))) # Use PIL to resize
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(rsize.astype('double'), retrain_model.predict,  top_labels=3, hide_color=0, num_samples=1000)
    return explanation
def interpret2(image):
    exp= explain2(image)
    temp_1, mask_1 = exp.get_image_and_mask(exp.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    temp_2, mask_2 = exp.get_image_and_mask(exp.top_labels[0], positive_only=False, num_features=10, hide_rest=False)

    ind =  exp.top_labels[0]
    dict_heatmap = dict(exp.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(exp.segments) 
   
    fig, (ax1, ax2, ax3, ax4 ) = plt.subplots(1, 4,  figsize=(15,15))   
    ax1.imshow(image)
    ax1.set_title('Original image')
    ax2.imshow(mark_boundaries(temp_1, mask_1))
    ax2.set_title('Superpixel boundaries')
    ax3.imshow(mark_boundaries(temp_2, mask_2))
    ax3.set_title('Superpixel boundaries')
    ax4.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    ax4.set_title('Heat Map')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()