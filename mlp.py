# %% [markdown]
# Imports

# %%
import os
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
import nltk
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import regex
import matplotlib.pyplot as plt
from copy import deepcopy
np.random.seed(1234)
tf.random.set_seed(1234)
nltk.download('stopwords')

# %% [markdown]
# Preprocessing

# %%
def remove_noise(line:str):
  line = line.replace("<br /><br />","") #gets rid of <br />
  line = regex.sub("[^a-z0-9 ]"," ",line) #get rid of all non lowercase/numerical values and replaces with a space
  line = regex.sub("[ ]{2,}"," ",line)   #replaces all occurances of multiple spaces with just one space
  return line


def get_data(file):
  stop_words = stopwords.words('english')
  sw = defaultdict(lambda:1)
  for word in stop_words:
    sw[word]=0
  f = open(file)
  ps = PorterStemmer()
  reviews = []
  sentiments = []
  for line in f.readlines():
    splitter = line.rindex(",")
    review,sentiment = (line[:splitter],line[splitter+1:-1])
    #make lower
    review = review.lower()
    #remove noise
    review = remove_noise(review)
    #stem and stopword removal
    review = [ps.stem(word) for word in review.split(" ") if sw[word]]
    review = " ".join(review)
    reviews.append(review[1:-1])#append tokenized data
    if(sentiment=='positive'):
      sentiments.append(1)
    else:
      sentiments.append(0)
    #sentiments.append(sentiment)
  return reviews[1:],sentiments[1:]

# %%
reviews,sentiments = get_data("Assignment_2_modified_ Dataset.csv")

# %%
print(reviews[1])
print(reviews[0])
print(sentiments[1])
print(sentiments[0])

# %% [markdown]
# Split into Train Val Test(80,10,10)

# %%
trainCut = int(len(reviews)*0.8)
valCut = int(len(reviews)*0.9)
train_data = reviews[0:trainCut]
train_labels = sentiments[0:trainCut]
val_data = reviews[trainCut:valCut]
val_labels = sentiments[trainCut:valCut]
test_data = reviews[valCut:]
test_labels= sentiments[valCut:]


# %%
tokenizer = keras.preprocessing.text.Tokenizer(filters="", lower=False, split=' ', oov_token='OOV')

# %%
tokenizer.fit_on_texts(train_data)

# %%
tokenizer.get_config()

# %%
print(f"Vocabulary size: {len(tokenizer.word_index)}")

# %% [markdown]
# Vectorization

# %%
print(train_data[:1])
X_train = tokenizer.texts_to_matrix(train_data)#, mode='tfidf')#TF-IDF Bag of Words vectorization
X_val = tokenizer.texts_to_matrix(val_data)#,mode="tfidf")
X_test = tokenizer.texts_to_matrix(test_data)#,mode='tfidf')

# https://numpy.org/doc/stable/user/basics.indexing.html

# %%
size_hidden1 = 64
size_hidden2 = 64
size_hidden3 = 64
size_output = 2

number_of_train_examples = X_train.shape[0]
print(X_train.shape)
number_of_test_examples = X_test.shape[0]

y_train = tf.keras.utils.to_categorical(train_labels, num_classes=2) # Other function is tf.one_hot(y_train,depth=10)
y_val = tf.keras.utils.to_categorical(val_labels, num_classes=2)
y_test = tf.keras.utils.to_categorical(test_labels, num_classes=2)


# %%
print(X_test.shape)

# %%
print(y_train.shape[1])

# %%
class MLP_1(object):
  def __init__(self, size_input, size_hidden1, size_hidden2, size_hidden3, size_hidden4, size_output, lr=0.01, seed=None, device=None):
    """
    size_input: int, size of input layer
    size_hidden1: int, size of the 1st hidden layer
    size_hidden2: int, size of the 2nd hidden layer
    size_hidden3: int, size of the 3rd hidden layer
    size_hidden4: int, size of the 4rth hidden layer
    size_output: int, size of output layer
    device: str or None, either 'cpu' or 'gpu' or None. If None, the device to be used will be decided automatically during Eager Execution
    """
    self.size_input, self.size_hidden1, self.size_hidden2, self.size_hidden3, self.size_hidden4, self.size_output, self.lr, self.seed, self.device =\
    size_input, size_hidden1, size_hidden2, size_hidden3, size_hidden4, size_output, lr, seed, device

    if(seed!=None):
      tf.random.set_seed(seed=seed)

    # Initialize weights between input mapping and a layer g(f(x)) = layer
    self.W_In = tf.Variable(tf.random.normal([self.size_input, self.size_hidden1],stddev=0.1)) # Xavier(Fan-in fan-out) and Orthogonal
    # Initialize biases for hidden layer
    self.b_In = tf.Variable(tf.zeros([1, self.size_hidden1])) # 0 or constant(0.01)

    # Initialize weights between input layer and 1st hidden layer
    self.W1 = tf.Variable(tf.random.normal([self.size_hidden1, self.size_hidden2],stddev=0.1))
    # Initialize biases for hidden layer
    self.b1 = tf.Variable(tf.zeros([1, self.size_hidden2]))

    # Initialize weights between input layer and 1st hidden layer
    self.W2 = tf.Variable(tf.random.normal([self.size_hidden2, self.size_hidden3],stddev=0.1))
    # Initialize biases for hidden layer
    self.b2 = tf.Variable(tf.zeros([1, self.size_hidden3]))

    # Initialize weights between input layer and 1st hidden layer
    self.W3 = tf.Variable(tf.random.normal([self.size_hidden3, self.size_hidden4],stddev=0.1))
    # Initialize biases for hidden layer
    self.b3 = tf.Variable(tf.zeros([1, self.size_hidden4]))

    # Initialize weights between 1st hidden layer and output layer
    self.W_Out = tf.Variable(tf.random.normal([self.size_hidden4, self.size_output],stddev=0.1))
    # Initialize biases for output layer
    self.b_Out = tf.Variable(tf.zeros([1, self.size_output]))

    # Define variables to be updated during backpropagation
    self.variables = [self.W_In, self.W1, self.W2, self.W3, self.W_Out, self.b_In, self.b1, self.b2, self.b3, self.b_Out]
    self.variable_history = [deepcopy(self.variables)]

  def forward(self, X):
    """
    forward pass
    X: Tensor, inputs
    """
    if self.device is not None:
      with tf.device('gpu:0' if self.device=='gpu' else 'cpu'):
        self.y = self.compute_output(X)
    else:
      self.y = self.compute_output(X)

    return self.y

  def loss(self, y_pred, y_true):
    '''
    y_pred - Tensor of shape (batch_size, size_output)
    y_true - Tensor of shape (batch_size, size_output)
    '''
    #y_true_tf = tf.cast(tf.reshape(y_true, (-1, self.size_output)), dtype=tf.float32)
    y_true_tf = tf.cast(y_true, dtype=tf.float32)
    y_pred_tf = tf.cast(y_pred, dtype=tf.float32)

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_x = cce(y_true_tf, y_pred_tf)
    # Use keras or tf_softmax, both should work for any given model
    #loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_tf, labels=y_true_tf))
    #L2 Regularization
    l2 = tf.reduce_sum([tf.nn.l2_loss(weight) for weight in self.variables])
    l2 = self.lr*tf.sqrt(l2)
    return loss_x+l2

  def backward(self, X_train, y_train,optimizer:tf.optimizers.Adam):
    """
    backward pass
    """


    with tf.GradientTape() as tape:

      predicted = self.forward(X_train)
      current_loss = self.loss(predicted, y_train)

    #print("pred",predicted)
    #print("loss",current_loss)
    #print(self.variables)
    grads = tape.gradient(current_loss, self.variables)
    optimizer.apply_gradients(zip(grads,self.variables))
    #print("grad",grads)
    return grads


  def compute_output(self, X):
    """
    Custom method to obtain output tensor during forward pass
    """
    # Cast X to float32
    X_tf = tf.cast(X, dtype=tf.float32)
    #X_tf = X

    # Compute values in hidden layers
    h_In = tf.matmul(X_tf, self.W_In) + self.b_In
    z_In = tf.nn.relu(h_In)

    h1 = tf.matmul(z_In, self.W1) + self.b1
    z1 = tf.nn.relu(h1)

    h2 = tf.matmul(z1, self.W2) + self.b2
    z2 = tf.nn.relu(h2)

    h3 = tf.matmul(z2, self.W3) + self.b3
    z3 = tf.nn.relu(h3)


    # Compute output
    output = tf.matmul(z3, self.W_Out) + self.b_Out

    #Now consider two things , First look at inbuild loss functions if they work with softmax or not and then change this
    # Second add tf.Softmax(output) and then return this variable
    return (output)



# %%
class MLP_2(object):
  def __init__(self, size_input, size_hidden1, size_hidden2, size_hidden3, size_hidden4,size_hidden5, size_output, lr=0.01, seed=None, device=None):
    """
    size_input: int, size of input layer
    size_hidden1: int, size of the 1st hidden layer
    size_hidden2: int, size of the 2nd hidden layer
    size_hidden3: int, size of the 3rd hidden layer
    size_hidden4: int, size of the 4rth hidden layer
    size_output: int, size of output layer
    device: str or None, either 'cpu' or 'gpu' or None. If None, the device to be used will be decided automatically during Eager Execution
    """
    self.size_input, self.size_hidden1, self.size_hidden2, self.size_hidden3, self.size_hidden4,self.size_hidden5, self.size_output, self.lr, self.seed, self.device =\
    size_input, size_hidden1, size_hidden2, size_hidden3, size_hidden4, size_hidden5, size_output, lr, seed, device

    if(seed!=None):
      tf.random.set_seed(seed=seed)

    # Initialize weights between input mapping and a layer g(f(x)) = layer
    self.W_In = tf.Variable(tf.random.normal([self.size_input, self.size_hidden1],stddev=0.1)) # Xavier(Fan-in fan-out) and Orthogonal
    # Initialize biases for hidden layer
    self.b_In = tf.Variable(tf.zeros([1, self.size_hidden1])) # 0 or constant(0.01)

    # Initialize weights between input layer and 1st hidden layer
    self.W1 = tf.Variable(tf.random.normal([self.size_hidden1, self.size_hidden2],stddev=0.1))
    # Initialize biases for hidden layer
    self.b1 = tf.Variable(tf.zeros([1, self.size_hidden2]))

    # Initialize weights between input layer and 1st hidden layer
    self.W2 = tf.Variable(tf.random.normal([self.size_hidden2, self.size_hidden3],stddev=0.1))
    # Initialize biases for hidden layer
    self.b2 = tf.Variable(tf.zeros([1, self.size_hidden3]))

    # Initialize weights between input layer and 1st hidden layer
    self.W3 = tf.Variable(tf.random.normal([self.size_hidden3, self.size_hidden4],stddev=0.1))
    # Initialize biases for hidden layer
    self.b3 = tf.Variable(tf.zeros([1, self.size_hidden4]))

    # Initialize weights between input layer and 1st hidden layer
    self.W4 = tf.Variable(tf.random.normal([self.size_hidden4, self.size_hidden5],stddev=0.1))
    # Initialize biases for hidden layer
    self.b4 = tf.Variable(tf.zeros([1, self.size_hidden5]))

    # Initialize weights between 1st hidden layer and output layer
    self.W_Out = tf.Variable(tf.random.normal([self.size_hidden5, self.size_output],stddev=0.1))
    # Initialize biases for output layer
    self.b_Out = tf.Variable(tf.zeros([1, self.size_output]))

    # Define variables to be updated during backpropagation
    self.variables = [self.W_In, self.W1, self.W2, self.W3, self.W4, self.W_Out, self.b_In, self.b1, self.b2, self.b3, self.b4, self.b_Out]
    self.variable_history = [deepcopy(self.variables)]

  def forward(self, X):
    """
    forward pass
    X: Tensor, inputs
    """
    if self.device is not None:
      with tf.device('gpu:0' if self.device=='gpu' else 'cpu'):
        self.y = self.compute_output(X)
    else:
      self.y = self.compute_output(X)

    return self.y

  def loss(self, y_pred, y_true):
    '''
    y_pred - Tensor of shape (batch_size, size_output)
    y_true - Tensor of shape (batch_size, size_output)
    '''
    #y_true_tf = tf.cast(tf.reshape(y_true, (-1, self.size_output)), dtype=tf.float32)
    y_true_tf = tf.cast(y_true, dtype=tf.float32)
    y_pred_tf = tf.cast(y_pred, dtype=tf.float32)

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_x = cce(y_true_tf, y_pred_tf)
    # Use keras or tf_softmax, both should work for any given model
    #loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_tf, labels=y_true_tf))

    #L2 Regularization
    l2 = tf.reduce_sum([tf.nn.l2_loss(weight) for weight in self.variables])
    l2 = self.lr*tf.sqrt(l2)
    return loss_x+l2

  def backward(self, X_train, y_train,optimizer:tf.optimizers.Adam):
    """
    backward pass
    """


    with tf.GradientTape() as tape:

      predicted = self.forward(X_train)
      current_loss = self.loss(predicted, y_train)

    #print("pred",predicted)
    #print("loss",current_loss)
    #print(self.variables)
    grads = tape.gradient(current_loss, self.variables)
    optimizer.apply_gradients(zip(grads,self.variables))
    #print("grad",grads)
    return grads


  def compute_output(self, X):
    """
    Custom method to obtain output tensor during forward pass
    """
    # Cast X to float32
    X_tf = tf.cast(X, dtype=tf.float32)
    #X_tf = X

    # Compute values in hidden layers
    h_In = tf.matmul(X_tf, self.W_In) + self.b_In
    z_In = tf.nn.relu(h_In)

    h1 = tf.matmul(z_In, self.W1) + self.b1
    z1 = tf.nn.relu(h1)

    h2 = tf.matmul(z1, self.W2) + self.b2
    z2 = tf.nn.relu(h2)

    h3 = tf.matmul(z2, self.W3) + self.b3
    z3 = tf.nn.relu(h3)

    h4 = tf.matmul(z3, self.W4) + self.b4
    z4 = tf.nn.relu(h4)

    # Compute output
    output = tf.matmul(z4, self.W_Out) + self.b_Out

    #Now consider two things , First look at inbuild loss functions if they work with softmax or not and then change this
    # Second add tf.Softmax(output) and then return this variable
    return (output)



# %% [markdown]
# Training multiple MLP with different Hyper Parameters

# %%
def train_mlp(mlp,batch_size=100,epochs=25,lr=1e-4,X_train=X_train,y_train=y_train,X_val=X_val,y_val=y_val):
  time_start = time.time()
  optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

  X_batches = [X_train[i:i+batch_size] for i in range(0, len(X_train), batch_size)]
  y_batches = [y_train[i:i+batch_size] for i in range(0, len(y_train), batch_size)]
  # accuracy_per_batch = []
  # loss_per_batch = []
  mlp_epoch_accuracies = []
  mlp_epoch_losses = []
  mlp_val_accuracies = []
  mlp_val_losses = []

  for epoch in range(epochs):
    ### Define your training loop
    for X_batch, y_batch in zip(X_batches, y_batches):
      mlp_accuracy_per_batch = []
      mlp_loss_per_batch = []

      mlp_output = mlp.forward(X_batch)

      mlp_loss_per_batch.append(mlp.loss(mlp_output,y_batch))#*len(y_batch))

      mlp_predictions = np.argmax(mlp_output, axis=1)

      y_true = np.argmax(y_batch, axis=1)
      mlp_accuracy_per_batch.append(mlp_predictions==y_true)

      dinput = mlp.backward(X_batch,y_batch,optimizer)#Update variables
      # for i in range(len(mlp1.variables)):
      #   mlp1.variables[i].assign_sub(dinput[i])
    mlp.variable_history.append(deepcopy(mlp.variables))#Record epoch weights when done

    mlp_epoch_accuracy = np.mean(mlp_accuracy_per_batch)
    mlp_epoch_loss = np.mean(mlp_loss_per_batch)

    mlp_epoch_accuracies.append(mlp_epoch_accuracy)
    mlp_epoch_losses.append(mlp_epoch_loss)

    mlp_y_val_preds = mlp.forward(X_val)
    mlp_val_predictions = np.argmax(mlp_y_val_preds, axis=1)

    y_val_true = np.argmax(y_val, axis=1)
    mlp_val_accuracy = np.mean((mlp_val_predictions==y_val_true))
    mlp_val_loss = np.mean(mlp.loss(mlp_y_val_preds,y_val))#*len(y_val)

    mlp_val_accuracies.append(mlp_val_accuracy)
    mlp_val_losses.append(mlp_val_loss)
    print(f'MLP - epoch: {epoch+1}, ' +
          f'training acc: {mlp_epoch_accuracy:.3f}, ' +
          f'training loss: {mlp_epoch_loss:.3f}, '+
          f'validation acc: {mlp_val_accuracy:.3f}, '+
          f'validation loss: {mlp_val_loss:.3f}')
    # if(mlp1_epoch_accuracy>0.9 and mlp2_epoch_accuracy>0.9 and mlp1_val_accuracy>0.87 and mlp2_val_accuracy>0.87):
    #   break
    # if(len(epoch_accuracies)>1 and epoch_losses[-1]>epoch_losses[-2] and max(epoch_accuracies)>0.8):
    #   print(epoch_losses[-1],epoch_losses[-2])
    #   break
  time_taken = time.time() - time_start
  print(f'time taken:{time_taken:.3f} seconds')
  return {"Epoch Acc":mlp_epoch_accuracies,"Epoch Losses":mlp_epoch_losses,"Val Acc":mlp_val_accuracies,"Val Losses": mlp_val_losses}


# %%
def plot_scores(scores):
  plt.figure(figsize=(16, 8))
  plt.subplot(1, 2, 1)
  plt.plot(scores['Epoch Acc'])
  plt.plot(scores['Val Acc'])
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend(["Epoch Acc", "Val Acc"])
  plt.subplot(1, 2, 2)
  plt.plot(scores['Epoch Losses'])
  plt.plot(scores['Val Losses'])
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend(["Epoch Losses", "Val Losses"])

# %% [markdown]
# Set Hyper Parameters and create MLPs

# %%
learning_rate = 0.5
# Initialize model using CPU
size_input = X_train.shape[1]
size_output = y_train.shape[1]
seed = 1111111#43770
mlp1 = MLP_1(size_input, 96, 96, 96, 96, size_output,learning_rate,seed=seed, device='gpu')
#scores1 = train_mlp(mlp1,500)
mlp2 = MLP_2(size_input, 96, 96, 96, 96, 96, size_output,learning_rate, seed=seed, device='gpu')
#scores2 = train_mlp(mlp2,500)

# %% [markdown]
# Train MLP 1

# %%
scores1 = train_mlp(mlp1,batch_size=500)

# %%
plot_scores(scores1)


# %% [markdown]
# Train MLP 2

# %%
scores2 = train_mlp(mlp2,batch_size=500)

# %%
plot_scores(scores2)

# %% [markdown]
# Test

# %%
def confusion_matrix(y_pred,y_test=y_test):
  tp = 0
  fp = 0
  fn = 0
  tn = 0
  for i in range(len(y_pred)):
    #print(y_pred[i][0],y_test[i][0])
    if(y_pred[i][0]>0 and y_test[i][0]==1):
      tp+=1
    elif(y_pred[i][0]<0 and y_test[i][0]==0):
      tn+=1
    elif(y_pred[i][0]>0 and y_test[i][0]==0):
      fp+=1
    elif(y_pred[i][0]<0 and y_test[i][0]==1):
      fn+=1
  print(f"\tTP:{tp}\tFP:{fp}")
  print(f"\tFN:{fn}\tTN:{tn}")
  #print((tp+tn)/(tp+tn+fp+fn))


# %%
mlp1_y_test_preds = mlp1.forward(X_test)
mlp2_y_test_preds = mlp2.forward(X_test)
mlp1_y_predictions = np.argmax(mlp1_y_test_preds, axis=1)
mlp2_y_predictions = np.argmax(mlp2_y_test_preds, axis=1)
y_true = np.argmax(y_test, axis=1)
mlp1_test_accuracy = (mlp1_y_predictions==y_true)
mlp2_test_accuracy = (mlp2_y_predictions==y_true)
print("MLP1 Test Accuracy: ",np.mean(mlp1_test_accuracy))
print("MLP1 Confusion Matrix: ")
confusion_matrix(mlp1_y_test_preds)
print("MLP2 Test Accuracy: ",np.mean(mlp2_test_accuracy))
print("MLP2 Confusion Matrix: ")
confusion_matrix(mlp2_y_test_preds)

# %% [markdown]
# Experimental Evaluation

# %%
mlp1_experiment = []
mlp2_experiment = []
y_true = np.argmax(y_test, axis=1)
for i in range(10):
  print(i)
  mlp1_exp = MLP_1(size_input, 96, 96, 96, 96, size_output,learning_rate,seed=i, device='gpu')
  mlp2_exp = MLP_2(size_input, 96, 96, 96, 96, 96, size_output,learning_rate, seed=i, device='gpu')
  train_mlp(mlp1_exp,batch_size=500)
  train_mlp(mlp2_exp,batch_size=500)
  mlp1_y_test_preds = mlp1_exp.forward(X_test)
  mlp2_y_test_preds = mlp2_exp.forward(X_test)
  mlp1_y_predictions = np.argmax(mlp1_y_test_preds, axis=1)
  mlp2_y_predictions = np.argmax(mlp2_y_test_preds, axis=1)
  mlp1_test_accuracy = (mlp1_y_predictions==y_true)
  mlp2_test_accuracy = (mlp2_y_predictions==y_true)
  mlp1_experiment.append(np.mean(mlp1_test_accuracy))
  mlp2_experiment.append(np.mean(mlp2_test_accuracy))
print("DONE RUNNING EXPERIMENT")

# %%
print("MLP1 With Different Seeds Ran 10 times: ",mlp1_experiment)
print(f"MLP1 Mean: {np.mean(mlp1_experiment):.3f}\t MLP1 Stdev: {np.std(mlp1_experiment):.4f}")
print("MLP2 With Different Seeds Ran 10 times: ",mlp2_experiment)
print(f"MLP2 Mean: {np.mean(mlp2_experiment):.3f}\t MLP2 Stdev: {np.std(mlp2_experiment):.4f}")


