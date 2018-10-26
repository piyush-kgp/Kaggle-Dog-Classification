
# coding: utf-8

# In[2]:


import tensorflow as tf
import tensorflow_hub as hub
import os
import glob
import cv2
import random
import pandas as pd
import numpy as np


# In[3]:


class model_config:
    learning_rate = 0.01
    batch_size = 100
    num_epochs = 20



# In[4]:




# In[5]:


def load_imgs(folder, module):
    expected_size = tuple(hub.get_expected_image_size(module))
    imgs_list = []
    id_list = []
    for file_name in glob.glob(folder+'/*.jpg'):
        idf = file_name.split('/')[-1][:-4]
        # print(idf)
        id_list.append(idf)
        img = cv2.imread(file_name)
        img = cv2.resize(img, expected_size)
        imgs_list.append(img)
    no_of_img = len(imgs_list)
    img_mat = np.stack(imgs_list, axis = 0)
    print('Images loaded for folder', folder)
    return no_of_img, img_mat, expected_size, id_list


# In[6]:


def load_label(csv_file, id_list, id_col_name = 'id', label_col_name = 'breed'):
    data = pd.read_csv(csv_file)
    classes = sorted(list(set(data[label_col_name])))
    # print(classes)
    num_classes = len(classes)
    y_s = []
    for id in id_list:
        y = np.zeros(num_classes)
        class_label = data.loc[data[id_col_name]==id, label_col_name].iloc[0]
        y[classes.index(class_label)] = 1
        y_s.append(y)
    print('Labels loaded for file', csv_file)
    return np.stack(y_s, axis = 0), classes, num_classes


# In[7]:




# In[8]:


def train_test_split(train_data, train_labels, train_val_split=0.99):
    train_size = train_data.shape[0]
    split = int(train_size*train_val_split)
    train_X = train_data[:split, :, :,:]
    train_Y = train_labels[:split, :]
    val_X = train_data[split:, :, :, :]
    val_Y = train_labels[split:, :]
    print('Train Val Split done..')
    return train_X, train_Y, val_X, val_Y


# In[9]:




# In[10]:


def create_batch(input_mat, batch_num, batch_size):
    return input_mat[batch_num*batch_size:(batch_num+1)*batch_size, ...]


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     X_train = create_batch(train_X, batch_num=0, batch_size=mymodel.batch_size)
#     Y_train = create_batch(train_Y, batch_num=0, batch_size=mymodel.batch_size)
#     _, batch_loss, batch_acc, pred_label, act_label = sess.run((train_op, loss, accuracy, predicted_label, real_label), feed_dict = {X_: X_train, Y_: Y_train})
#     print(_, batch_loss, batch_acc, pred_label, act_label)
#     X_train = create_batch(train_X, batch_num=1, batch_size=mymodel.batch_size)
#     Y_train = create_batch(train_Y, batch_num=1, batch_size=mymodel.batch_size)
#     _, batch_loss, batch_acc, pred_label, act_label = sess.run((train_op, loss, accuracy, predicted_label, real_label), feed_dict = {X_: X_train, Y_: Y_train})
#     print(_, batch_loss, batch_acc, pred_label, act_label)


# In[19]:


def network(X_, Y_, module, model):
    features = module(X_)
    num_classes = int(Y_.get_shape()[-1])
    logits = tf.layers.dense(inputs=features, units=num_classes)
    prediction = tf.nn.softmax(logits)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=model.learning_rate)
    train_op = optimizer.minimize(loss)
    
    predicted_label = tf.argmax(prediction, axis = 1)
    real_label = tf.argmax(Y_, axis = 1)

    correct_pred = tf.equal(predicted_label, real_label)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    return train_op, loss, accuracy, predicted_label, real_label



def train(train_X, train_Y, val_X, val_Y, model, module, train_classes, test_data, test_ids):
    saver = tf.train.Saver(max_to_keep = 0)
    batch_size = model.batch_size
    num_epochs = model.num_epochs
    req_shape = train_X.shape[1:]
    req_shape_Y = train_Y.shape[1:]
    num_batches = int(np.ceil(train_X.shape[0]/batch_size))

    X_ = tf.placeholder(tf.float32, (None, req_shape[0], req_shape[1], req_shape[2]))
    Y_ = tf.placeholder(tf.float32, (None, req_shape_Y[0]))

    features = module(X_)
    logits = tf.layers.dense(inputs=features, units=120)
    prediction = tf.nn.softmax(logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=mymodel.learning_rate)
    train_op = optimizer.minimize(loss)

    predicted_label = tf.argmax(prediction, axis = 1)
    real_label = tf.argmax(Y_, axis = 1)

    correct_pred = tf.equal(predicted_label, real_label)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    ctr = 0
    train_classes.insert(0, 'id')

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            for batch_num in range(num_batches):
                
                ctr+=1
                print('Session running at counter', ctr)
                X_train = create_batch(train_X, batch_num, batch_size)
                Y_train = create_batch(train_Y, batch_num, batch_size)
                
                results = train_op, loss, accuracy, predicted_label, real_label
                _, batch_loss, batch_acc, batch_pred, batch_real = sess.run(results, feed_dict = {X_: X_train, Y_: Y_train})
                _, val_loss, val_acc, val_pred, val_real = sess.run(results, feed_dict = {X_: val_X, Y_: val_Y})

                print('Epoch', epoch, 'Batch', batch_num, 'Batch Acc', 100*batch_acc, 'Val Acc', 100*val_acc)
                saver.save(sess, os.path.join(os.getcwd(), 'ckpoints'), global_step=10)
                if batch_num % 50==0:
                    num_batches_test = int(np.ceil(test_data.shape[0]/batch_size))
                    results_prob = []
                    results_label = []
                    for part in range(num_batches_test):
                        X_test = create_batch(test_data, part, batch_size)
                        test_pred, test_pred_labels = sess.run((prediction, predicted_label), feed_dict = {X_: X_test})
                        results_prob.append(test_pred)
                        results_label.append(test_pred_labels)
                        print(part, 'Done!!')

                    test_data_prediction = np.concatenate(results_prob, axis  = 0)
                    test_data_predict_label = np.concatenate(results_label, axis = 0)

                    to_write_subm = pd.DataFrame(columns = train_classes)
                    to_write_subm.id = test_ids

                    for i in range(len(test_ids)):
                        to_write_subm.loc[i, :][1:] = test_data_prediction[i, :]

                    to_write_subm.to_csv('subm_{}.csv'.format(ctr), index=False)


                    predicted_labels = [train_classes[x+1] for x in test_data_predict_label]
                    to_write_label = pd.DataFrame({'id': test_ids, 'Predicted Breed': predicted_labels})
                    to_write_label.to_csv('subm_label_{}.csv'.format(ctr), index = False)

                
            print('-------------------------------------------------------------------Epoch', epoch)
    return val_loss, val_acc, val_pred, val_real


# In[ ]:
if __name__ == '__main__':
    mymodel = model_config()
    inception_v3 = hub.Module('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')
    train_folder = 'train_data/train'
    test_folder = 'test_data/test'

    no_of_imgs_in_train_data, train_data, image_size, train_ids = load_imgs(folder = train_folder, module = inception_v3)
    no_of_imgs_in_test_data, test_data, _, test_ids = load_imgs(folder = test_folder, module = inception_v3)
    train_labels, train_classes, num_train_classes = load_label('all/labels.csv', train_ids, id_col_name = 'id', label_col_name = 'breed')

    train_X, train_Y, val_X, val_Y = train_test_split(train_data, train_labels, train_val_split=0.99)
    train(train_X = train_X, train_Y = train_Y, val_X = val_X, val_Y = val_Y, model=mymodel, module=inception_v3, train_classes = train_classes, test_data = test_data, test_ids = test_ids)
