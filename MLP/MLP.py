"""Libraries"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

"""Creat train, validation and test datasets"""
################################## Creat train and validation dataset #######################################
DataTrain = np.genfromtxt('./train.csv', delimiter=',',dtype = np.uint8)

# Samples
SampleTrain = (DataTrain[1:,1:]/255).astype(np.float32)

# Labels
LabelTrain = DataTrain[1:,0]
LabelTrain[LabelTrain > 9] -= 1

# Make onehot labels
LB = LabelBinarizer()
LabelTrainOnehot = LB.fit_transform(LabelTrain)

# Split train and validation datasets
X_train, X_valid, y_train, y_valid = train_test_split(SampleTrain, LabelTrainOnehot, test_size=0.3, random_state=42)


######################################## Creat test dataset ################################################
DataTest = np.genfromtxt('./test.csv', delimiter=',',dtype = np.uint8)

# Samples
SampleTest = (DataTest[1:,1:]/255).astype(np.float32)
X_test = SampleTest;

# Labels
LabelTest = DataTest[1:,0]
LabelTest[LabelTest > 9] -= 1

# Make onehot labels
LB = LabelBinarizer()
LabelTestOnehot = LB.fit_transform(LabelTest)
y_test = LabelTestOnehot;

"""
**Uncomment this part to see dimensions of datasets
######################################### Verify dimensions ###############################################
print('DIMENSIONS:')
print(f'SampleTrain       Shape: {SampleTrain.shape}  Type: {SampleTrain.dtype}')
print(f'LabelTrainOnehot  Shape: {LabelTrainOnehot.shape}   Type: {LabelTrainOnehot.dtype}')
print(f'X_train           Shape: {X_train.shape}  Type: {X_train.dtype}')
print(f'y_train           Shape: {y_train.shape}   Type: {y_train.dtype}')
print(f'X_valid           Shape: {X_valid.shape}   Type: {X_valid.dtype}')
print(f'y_valid           Shape: {y_valid.shape}    Type: {y_valid.dtype}')
print(f'SampleTest        Shape: {SampleTest.shape}   Type: {SampleTest.dtype}')
print(f'LabelTestOnehot   Shape: {LabelTestOnehot.shape}    Type: {LabelTestOnehot.dtype}')
"""

"""
**Uncomment this part to see 25 sample of training dataset
titles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

fig, axs = plt.subplots(5, 5, figsize=(10, 10));
for i in range(5):
    for j in range(5):
        axs[i, j].imshow(np.reshape(SampleTrain[5*i+j],(28,-1)),cmap='gray');
        axs[i, j].axis('off')
        axs[i, j].set_title(f'{titles[LabelTrain[5*i+j].astype(np.uint8)]}')
"""

"""Network artchitecture"""
tf.reset_default_graph()

############################################### Hyperparameters #################################################

num_output = 24
num_layers_1 = 784
num_layers_2 = 784
num_layers_3 = 256
learning_rate = 0.001

################################################### weights #####################################################

w1 = tf.get_variable(name='weight_1', shape=(784,num_layers_1),
                     initializer=tf.initializers.glorot_normal, dtype=tf.float32, trainable=True)

b1 = tf.get_variable(name='bias_1', shape=(num_layers_1), initializer=tf.zeros, dtype=tf.float32, trainable=True)

w2 = tf.get_variable(name='weight_2', shape=(num_layers_1,num_layers_2),
                     initializer=tf.initializers.glorot_normal, dtype=tf.float32, trainable=True)

b2 = tf.get_variable(name='bias_2', shape=(num_layers_2), initializer=tf.zeros, dtype=tf.float32, trainable=True)

w3 = tf.get_variable(name='weight_3', shape=(num_layers_2,num_layers_3),
                     initializer=tf.initializers.glorot_normal, dtype=tf.float32, trainable=True)

b3 = tf.get_variable(name='bias_3', shape=(num_layers_3), initializer=tf.zeros, dtype=tf.float32, trainable=True)

w4 = tf.get_variable(name='weight_4', shape=(num_layers_3,num_output),
                     initializer=tf.initializers.glorot_normal, dtype=tf.float32, trainable=True)

b4 = tf.get_variable(name='bias_4', shape=(num_output), initializer=tf.zeros, dtype=tf.float32, trainable=True)

############################################## architecture #####################################################

InputLayer = tf.placeholder(shape=(None,784), dtype=tf.float32, name='InputLayer')

InputY = tf.placeholder(shape=(None,num_output), dtype=tf.float32, name='InputY')

Output1 = tf.nn.tanh(tf.matmul(InputLayer,w1)+b1, name='Output1')

Output2 = tf.nn.tanh(tf.matmul(Output1,w2)+b2, name='Output2')

Output3 = tf.nn.tanh(tf.matmul(Output2,w3)+b3, name='Output3')

Output4 = tf.nn.tanh(tf.matmul(Output3,w4)+b4, name='Output4')

OutputLayer = tf.nn.softmax(Output4, name='OutputLayer')

########################################### Training Part ######################################################

Predicted = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Output4, labels=InputY)

Loss = tf.reduce_mean(Predicted)

LearningeRate = tf.train.exponential_decay(learning_rate, 0, 15, 0.80, staircase=True)

TrainSGD = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(Loss)

TrainAdam = tf.train.AdamOptimizer(learning_rate=LearningeRate).minimize(Loss)

####################################### Tensorboard variables ##################################################

loss_train=tf.summary.scalar('TrainingLoss', Loss)
loss_valid=tf.summary.scalar('ValidationLoss', Loss)
loss_test=tf.summary.scalar('TestLoss', Loss)

bias1=tf.summary.histogram ('bias1', b1)
bias2=tf.summary.histogram ('bias2', b2)
bias3=tf.summary.histogram ('bias3', b3)
bias4=tf.summary.histogram ('bias4', b4)

weight1=tf.summary.histogram ('weight1', w1)
weight2=tf.summary.histogram ('weight2', w2)
weight3=tf.summary.histogram ('weight3', w3)
weight4=tf.summary.histogram ('weight4', w4)

"""Train Model"""
############################################### Hyperparameters #################################################

BatchSize = 512
Epoch = 100

################################################## Training #####################################################

NumberBatch = X_train.shape[0]//BatchSize

with tf.Session() as sess:
    write = tf.summary.FileWriter('./graphs', sess.graph)

    sess.run(tf.global_variables_initializer())

    TestAcc=100*accuracy_score(y_test.argmax(1), sess.run(OutputLayer, {InputLayer:X_test}).argmax(1))
    TrainAcc=100*accuracy_score(y_train.argmax(1), sess.run(OutputLayer, {InputLayer:X_train}).argmax(1))
    ValidationAcc=100*accuracy_score(y_valid.argmax(1), sess.run(OutputLayer, {InputLayer:X_valid}).argmax(1))
    print('Epoch {0}: training:{1:.3f}    valid:{2:.3f}    test:{3:.3f} '.format(0, TrainAcc ,ValidationAcc ,TestAcc))

    write.add_summary(sess.run(loss_train, feed_dict={InputLayer:X_train, InputY:y_train}), 0)
    write.add_summary(sess.run(loss_test, feed_dict={InputLayer:X_test, InputY:y_test}), 0)
    write.add_summary(sess.run(loss_valid, feed_dict={InputLayer:X_valid, InputY:y_valid}), 0)

    write.add_summary(sess.run(bias1), 0)
    write.add_summary(sess.run(bias2), 0)
    write.add_summary(sess.run(bias3), 0)
    write.add_summary(sess.run(bias4), 0)

    write.add_summary(sess.run(weight1), 0)
    write.add_summary(sess.run(weight2), 0)
    write.add_summary(sess.run(weight3), 0)
    write.add_summary(sess.run(weight4), 0)

    for i in range(1, Epoch+1):
      arr = np.arange(X_train.shape[0])
      np.random.shuffle(arr)

      for j in range(NumberBatch+1):
        if j<NumberBatch:
          X_batch = X_train[arr[j*BatchSize:(1+j)*BatchSize]]
          y_batch = y_train[arr[j*BatchSize:(1+j)*BatchSize]]

        if j==NumberBatch:
          X_batch = X_train[arr[j*BatchSize:]]
          y_batch = y_train[arr[j*BatchSize:]] 

        sess.run(TrainAdam, feed_dict={InputLayer:X_batch, InputY:y_batch})
      if i%10 == 0:
        TestAcc=100*accuracy_score(y_test.argmax(1), sess.run(OutputLayer, {InputLayer:X_test}).argmax(1))
        TrainAcc=100*accuracy_score(y_train.argmax(1), sess.run(OutputLayer, {InputLayer:X_train}).argmax(1))
        ValidationAcc=100*accuracy_score(y_valid.argmax(1), sess.run(OutputLayer, {InputLayer:X_valid}).argmax(1))
        print('Epoch {0}: training:{1:.3f}    valid:{2:.3f}    test:{3:.3f} '.format(i, TrainAcc ,ValidationAcc ,TestAcc))
      
        write.add_summary(sess.run(loss_train, feed_dict={InputLayer:X_train, InputY:y_train}), i)
        write.add_summary(sess.run(loss_test, feed_dict={InputLayer:X_test, InputY:y_test}), i)
        write.add_summary(sess.run(loss_valid, feed_dict={InputLayer:X_valid, InputY:y_valid}), i)

        write.add_summary(sess.run(bias1), i)
        write.add_summary(sess.run(bias2), i)
        write.add_summary(sess.run(bias3), i)
        write.add_summary(sess.run(bias4), i)

        write.add_summary(sess.run(weight1), i)
        write.add_summary(sess.run(weight2), i)
        write.add_summary(sess.run(weight3), i)
        write.add_summary(sess.run(weight4), i)
        