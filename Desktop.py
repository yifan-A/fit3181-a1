#!/usr/bin/env python
# coding: utf-8

# # <span style="color:#0b486b">  FIT3181: Deep Learning (2022)</span>
# ***
# *CE/Lecturer:* Dr **Trung Le** | trunglm@monash.edu <br/>
# *Head Tutor:* Mr **Thanh Nguyen** | thanh.nguyen4@monash.edu  <br/>
# <br/>
# Department of Data Science and AI, Faculty of Information Technology, Monash University, Australia
# ***

# # <span style="color:#0b486b">  Student Information</span>
# ***
# Surname: **Dai**  <br/>
# Firstname: **Yifan**    <br/>
# Student ID: **31510477**    <br/>
# Email: **ydai0019@student.monash.edu**    <br/>
# Your tutorial time: **THU 10-12 a.m.**    <br/>
# ***

# # <span style="color:#0b486b">Deep Neural Networks</span>
# ### Due: <span style="color:red">11:59pm Sunday, 18 September 2022</span>  (Sunday)
# 
# #### <span style="color:red">Important note:</span> This is an **individual** assignment. It contributes **20%** to your final mark. Read the assignment instruction carefully.

# ## <span style="color:#0b486b">Instruction</span>
# 
# This notebook has been prepared for your to complete Assignment 1. The theme of this assignment is about practical machine learning knowledge and skills in deep neural networks, including feedforward and convolutional neural networks. Some sections have been partially completed to help you get
# started. **The total marks for this notebook is 100**.
# 
# * Before you start, read the entire notebook carefully once to understand what you need to do. <br/>
# 
# * For each cell marked with **#YOU ARE REQUIRED TO INSERT YOUR CODES IN THIS CELL**, there will be places where you **must** supply your own codes when instructed. <br>
# 
# This assignment contains **three** parts:
# 
# * Part 1: Questions on theory and knowledge on machine learning and deep learning **[30 points], 30%**
# * Part 2: Coding assessment on TensorFlow for Deep Neural Networks (DNN) **[30 points], 30%**
# * Part 3: Coding assessment on TensorFlow for Convolution Neural Networks (CNN) **[40 points], 40%**
# 
# **Hint**: This assignment was essentially designed based on the lectures and tutorials sessions covered from Week 1 to Week 6. You are strongly recommended to go through these contents thoroughly which might help you to complete this assignment.

# ## <span style="color:#0b486b">What to submit</span>
# 
# This assignment is to be completed individually and submitted to Moodle unit site. **By the due date, you are required to submit one  <span style="color:red; font-weight:bold">single zip file, named xxx_assignment01_solution.zip</span> where `xxx` is your student ID, to the corresponding Assignment (Dropbox) in Moodle**. 

# ***For example, if your student ID is <span style="color:red; font-weight:bold">12356</span>, then gather all of your assignment solution to folder, create a zip file named <span style="color:red; font-weight:bold">123456_assignment01_solution.zip</span> and submit this file.***

# Within this zip folder, you **must** submit the following files:
# 1.	**Assignment01_solution.ipynb**:  this is your Python notebook solution source file.
# 1.	**Assignment01_output.html**: this is the output of your Python notebook solution *exported* in html format.
# 1.	Any **extra files or folder** needed to complete your assignment (e.g., images used in your answers).

# Since the notebook is quite big to load and work together, one recommended option is to split solution into three parts and work on them seperately. In that case, replace **Assignment01_solution.ipynb** by three notebooks: **Assignment01_Part1_solution.ipynb**, **Assignment01_Part2_solution.ipynb** and **Assignment01_Part3_solution.ipynb**

# **You can run your codes on Google Colab. In this case, you need to capture the screenshots of your Google Colab model training and put in corresponding places in your Jupyter notebook. You also need to store your trained models to folder <span style="color:red; font-weight:bold">*./models*</span> with recognizable file names (e.g., Part3_Sec3_2_model.h5).** 

# ## <span style="color:#0b486b">Part 1: Theory and Knowledge Questions</span>
# <div style="text-align: right"><span style="color:red; font-weight:bold">[Total marks for this part: 30 points]<span></div>

# The first part of this assignment is for you to demonstrate your knowledge in deep learning that you have acquired from the lectures and tutorials materials. Most of the contents in this assignment are drawn from **the lectures and tutorials from weeks 1 to 3**. Going through these materials before attempting this part is highly recommended.

# ####  <span style="color:red">**Question 1.1**</span> **Activation function plays an important role in modern Deep NNs. For each of the activation function below, state its output range, find its derivative (show your steps), and plot the activation fuction and its derivative**
# 
# <span style="color:red">**(a)**</span> Leaky ReLU: $\text{LeakyReLU}\left(x\right)=\begin{cases}
# 0.01x & \text{if}\,x<0\\
# x & \text{otherwise}
# \end{cases}$ 
# <div style="text-align: right"><span style="color:red">[1.5 points]</span></div> 
# 
# <span style="color:red">**(b)**</span> Softplus: $\text{Softplus}\left(x\right)=\text{ln}\left(1+e^{x}\right)$
# <div style="text-align: right"><span style="color:red">[1.5 points]</span></div> 

# *(a) Leaky ReLU:*
# 
# *output range:*
# 
# $ \left ( -\infty ,+\infty \right ) $
# 
# *derivative:*
# 
#  $\text{LeakyReLU}\left(x\right)=\begin{cases}
# 0.01x & \text{for}\,x<0\\
# x & \text{otherwise}
# \end{cases}$ 
# 
# $\text{max}\left(0.01x,x\right)=\begin{cases}
# 0.01x & \text{for}\,x<0\\
# x & \text{otherwise}
# \end{cases}$ 
# 
# $\text{d/dx max}\left(0.01x,x\right)= \text {d/dx}\begin{cases}
# 0.01x & \text{for}\,x<0\\
# x & \text{otherwise}
# \end{cases}$ 
# 
# $\text{ d/dx max}\left(0.01x,x\right)= \begin{cases}
# 0.01 & \text{for}\,x<0\\
# 1 & \text{otherwise}
# \end{cases}$ 
# 

# In[7]:


import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from math import exp


# In[4]:


x = np.arange(-100, 100, 0.1)
y1 = np.where(x < 0, 0.01*x, x)
y2 = np.where(x < 0, 0.01, 1)  
plt.xlim(-5, 5)
plt.ylim(-2, 2)
plt.plot(x, y1, color="olive")
plt.plot(x, y2, color="orangered")


# *(b) SoftPlus*
# 
# *output range :*
# $ \left [ 0,\infty \right ) $
# 
# *Derivation :*
# 
# $\text{Softplus}\left(x\right)=\text{ln}\left(1+e^{x}\right)$
# 
# $\text{d/dx(Softplus}\left(x\right))=\text{d/dx (ln}\left(1+e^{x}\right))$
# 
# $=\left(1/ (1+e^{x}) * \text{d/dx(}1+e^{x})\right)$
# 
# $=\left((1/ (1+e^{x})) * e^{x}\right)$
# 
# $=\left(e^{x}/ (1+e^{x})\right)$
# 

# In[52]:


y3 = np.log(1 + np.exp(x))   
y4 = math.e ** (x) / (1+math.e ** (x))
plt.xlim(-5, 5)
plt.ylim(-2, 2)
plt.plot(x, y3,color="dimgrey")
plt.plot(x, y4,color="rosybrown")


# <span style="color:#0b486b"> **Numpy is possibly being used in the following questions. You need to import numpy here.** </span>

# In[6]:


import numpy as np


# ####  <span style="color:red">**Question 1.2**</span> **Assume that we feed a data point $x$ with a ground-truth label $y=2$ to the feed-forward neural network with the ReLU activation function as shown in the following figure**
# <img src="Figures/Q2_P1.png" width="500" align="center"/>
# 
# 
# <span style="color:red">**(a)**</span>  What is the numerical value of the latent presentation $h^1(x)$?
# <div style="text-align: right"><span style="color:red">[1 point]</span></div> 
# 
# <span style="color:red">**(b)**</span>  What is the numerical value of the latent presentation $h^2(x)$?
# <div style="text-align: right"><span style="color:red">[1 point]</span></div> 
# 
# <span style="color:red">**(c)**</span>  What is the numerical value of the logit $h^3(x)$?
# <div style="text-align: right"><span style="color:red">[1 point]</span></div> 
# 
# 
# <span style="color:red">**(d)**</span>  What is the corresonding prediction probabilities $p(x)$?
# <div style="text-align: right"><span style="color:red">[1 point]</span></div> 
# 
# <span style="color:red">**(e)**</span>  What is the cross-entropy loss caused by the feed-forward neural network at $(x,y)$? Remind that $y=2$.
# <div style="text-align: right"><span style="color:red">[1 point]</span></div> 
# 
# <span style="color:red">**(e)**</span>  Assume that we are applying the label smoothing technique (i.e.,  [link for main paper](https://papers.nips.cc/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf) from Goeff Hinton) with $\alpha = 0.1$. What is the relevant loss caused by the feed-forward neural network at $(x,y)$?
# <div style="text-align: right"><span style="color:red">[1 point]</span></div> 
# 
# 
# **You need to show both formulas and numerical results for earning full mark. Although it is optional, it is great if you show your numpy code for your computation.**

# *1.2(a)*
# 
# $\bar h^1$  = W1(x) + b1
#        
#        
# $h^1$ = relu($\bar h^1$)
#     
#     =  [ [4] , [1] , [6] , [2] ] 

# *1.2(b)*
# 
# $\bar h^2$ = W2(h1) + b2
# 
#        
# $h^2$ = relu($\bar h^2$)
#     
#     = [ [2] , [8] , [0] ]

# *1.2(c)*
# 
# 
# $h^3$ = W3($h^2$) + b3
#             
#        = [ [-14] , [ 18] , [6] ]
#        

# *1.2(d)*
# 
# p = softmax($h^3$)
# 
#     = [ [1.26640877e-14],
#        [9.99993856e-01],
#        [6.14417460e-06]] ]

# *1.2(e)*
# 
# cross entropy loss = - [ log2(1.26640877e-14) * 0 + log2(9.99993856e-01) * 1 + log2(6.14417460e-06) * 0 ]
# 
#                    = - [ log2(9.99993856e-01) ]
#                    
#                    = - [ 8.863945561269833e-06]              

# *1.2(f)*
# 
# the relevant loss caused by the feed-forward neural network at $(x,y)$ is -  8.863945561269833e-06

# ####  <span style="color:red">**Question 1.3**</span> **Assume that we are constructing a multilayered feed-forward neural network for a classification problem with three classes where the model parameters will be generated randomly using your student ID. The architecture of this network is ($3 (Input)\rightarrow4(LeakyReLU)\rightarrow 3(Output)$) as shown in the following figure. Note that the LeakyReLU has the same formula as the one in Q1.1.**
# 
# 
# <img src="Figures/Q3_P1.png" width="500" align="center"/>
# 
# We feed a feature vector $x=\left[\begin{array}{ccc}
# 1 & -1 & 1.5\end{array}\right]^{T}$ with ground-truth label $y=3$ to the above network. 
# 

# **You need to show both formulas, numerical results, and your numpy code for your computation for earning full marks.**

# In[1]:


#Code to generate random matrices and biases for W1, b1, W2, b2
import numpy as np
student_id = 31510477          #insert your student id here for example 1234    
np.random.seed(student_id)
W1 = np.random.rand(4,3)
b1 = np.random.rand(4,1)
W2 = np.random.rand(3,4)
b2 = np.random.rand(3,1)


# **Forward propagation**
# 
# <span style="color:red">**(a)**</span>  What is the value of $\bar{h}^{1}(x)$?
# <div style="text-align: right"><span style="color:red">[1 point]</span></div>
# 

# *Show your fomular*
# 
# **(a)**
# 
#  $\bar{h}^{1}(x)$ = W1 * x + b1
#                                                                                       

# In[2]:


# Show your code
x = np.array([ [1] , [-1] , [1.5] ])
h_bar1 = W1.dot(x)+b1
h_bar1


# <span style="color:red">**(b)**</span>  What is the value of $h^{1}(x)$?
# <div style="text-align: right"><span style="color:red">[1 point]</span></div>

# *Show your fomular*
# 
# $h^{1}(x)$ = leaky_relu ( $\bar{h}^{1}(x)$)
# 

# In[3]:


#Show your code
def leaky_relu(x,a=0.01):
    return np.maximum(a*x,x)

h1 = leaky_relu(h_bar1)
h1


# <span style="color:red">**(c)**</span>  What is the predicted value $\hat{y}$?
# <div style="text-align: right"><span style="color:red">[1 point]</span></div>

# *Show your fomular*
# 
# **(c)**
# 
# predicted value = output = $h^{2}(x)$
# 
# $\bar{h}^{2}(x)$ = W2 *  $h^{1}(x)$ + b2
#                 
# 
# $h^{2}(x)$ = leaky_relu($\bar{h}^{2}(x)$ )
# 
# 

# In[4]:


#Show your code

h_bar2 = W2.dot(h1)+b2
h2 = leaky_relu(h_bar2)
h2


# <span style="color:red">**(d)**</span>  Suppose that we use the cross-entropy (CE) loss. What is the value of the CE loss $l$?
# <div style="text-align: right"><span style="color:red">[1 point]</span></div>

# *Show your fomular*
# 
# **(d)**
# 
# Obtain the probability using softmax of the logit to calculate the CE Loss
# 
# logit = $\bar{h}^{2}(x)$
# 
# p = softmax(logit)
# 
#   = [0.19439049],
#    [0.4303551 ],
#    [0.37525441]]
#    
# q = [ 0 , 0 , 1 ]  -- ground-truth label  𝑦=3  
# 
# 
# CE_loss = - [ log2(0.19439049) * 0 + log2(0.4303551) * 0 + log2(0.37525441) * 1 ]
# 

# In[5]:


#Show your code
def softmax(x):
    
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x
logit = h_bar2
p = softmax(logit)
p


# In[11]:


ce_loss = - (np.log2(0.37525441))
ce_loss


# **Backward propagation**
# 
# <span style="color:red">**(e)**</span> What are the derivatives $\frac{\partial l}{\partial h^{2}},\frac{\partial l}{\partial W^{2}}$, and $\frac{\partial l}{\partial b^{2}}$? 
# <div style="text-align: right"><span style="color:red">[6 points]</span></div>

# $\frac{\partial l}{\partial h^{2}}$
# 
# 
# <div> = - y + softmax($h^{2}$) <div>
# <div> = $p^{T}$ - y <div> 
# 

# In[20]:


pt = np.transpose(p)
y = [0,0,1]
d_h2 = pt - y
d_h2


# $\frac{\partial l}{\partial W^{2}}$ = $\frac{\partial l}{\partial h^{2}}$ . $\frac{\partial h^{2}}{\partial W^{2}}$
# 

# In[21]:


d_h2_t = np.transpose(d_h2)

h1_t = np.transpose(h1)

d_w2 = d_h2_t.dot(h1_t)
d_w2


# 
# 
# $\frac{\partial l}{\partial b^{2}}$ = $\frac{\partial l}{\partial h^{2}}$ . $\frac{\partial h^{2}}{\partial b^{2}}$ 
#      

# In[22]:


d_h2


# <span style="color:red">**(f)**</span> What are the derivatives $\frac{\partial l}{\partial h^{1}}, \frac{\partial l}{\partial \bar{h}^{1}},\frac{\partial l}{\partial W^{1}}$, and $\frac{\partial l}{\partial b^{1}}$? 
# <div style="text-align: right"><span style="color:red">[6 points]</span></div>

#  $\frac{\partial l}{\partial h^{1}}$
# 
# $\frac{\partial l}{\partial h^{1}}$ = $\frac{\partial l}{\partial h^{2}}$ . $\frac{\partial h^{2}}{\partial h^{1}}$
# 
# =  $g^{2}$ . $W^{2}$ 

# In[23]:


d_h1 = d_h2.dot(W2)
d_h1 


# $\frac{\partial l}{\partial \bar h^{1}}$
# 
# $\frac{\partial l}{\partial \bar h^{1}}$  = $\frac{\partial l}{\partial h^{1}}$ . $\frac{\partial h^{1}}{\partial\bar h^{1}}$
#  
#  = $\frac{\partial l}{\partial h^{1}}$ * $\frac{\partial h^{1}}{\partial \bar h^{1}}$
#  
#  = $\frac{\partial l}{\partial h^{1}}$ * diag(d_leaky_relu( $\bar h^{1}$ ))

# In[24]:


np.diag(np.transpose(d_leaky_relu(h_bar1))[0])


# In[25]:


def d_leaky_relu(x, a=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = a
    return dx

d_h_bar2 = np.diag(tf.transpose(d_leaky_relu(h_bar1))[0])
d_h_bar1 = d_h1.dot(d_h_bar2)
d_h_bar1


#  $\frac{\partial l}{\partial W^{1}}$
# 
# $\frac{\partial l}{\partial W^{1}}$ = $\frac{\partial l}{\partial \bar h^{1}}^{T}$ . $h^{0^T}$
# 

# In[26]:


d_h_bar1_t = np.transpose(d_h_bar1)
d_h_bar1_t


# In[27]:


xt = np.transpose(x)
xt


# In[28]:


d_w1 = d_h_bar1_t.dot(xt)
d_w1


# **SGD update**
# 
# <span style="color:red">**(g)**</span> Assume that we use SGD with learning rate $\eta=0.01$ to update the model parameters. What are the values of $W^2, b^2$ and $W^1, b^1$ after updating?
# <div style="text-align: right"><span style="color:red">[5 points]</span></div>

# *Show your fomular*
# 
# $W = W - \eta*\frac{\partial l}{\partial W}$ 
# 
# $b^{1}$ and $b^{2} remain   the   same $ 

# In[36]:


#Show your code
l_r = 0.01
W11 = W1 - l_r
W22 = W2 - l_r


# In[31]:


W11


# In[32]:


W22


# In[34]:


b1


# In[35]:


b2


# ## <span style="color:#0b486b">Part 2: Deep Neural Networks (DNN) </span>
# <div style="text-align: right"><span style="color:red; font-weight:bold">[Total marks for this part: 30 points]<span></div>
# 
# The first part of this assignment is for you to demonstrate your basis knowledge in deep learning that you have acquired from the lectures and tutorials materials. Most of the contents in this assignment are drawn from **the tutorials covered from weeks 1 to 4**. Going through these materials before attempting this assignment is highly recommended.

# In the first part of this assignment, you are going to work with the **FashionMNIST** dataset for *image recognition task*. It has the exact same format as MNIST (70,000 grayscale images of 28 × 28 pixels each with 10 classes), but the images represent fashion items rather than handwritten digits, so each class is more diverse, and the problem is significantly more challenging than MNIST.

# ####  <span style="color:red">**Question 2.1**</span>. Load the Fashion MNIST using Keras datasets
# 
# <div style="text-align: right"> <span style="color:red">[5 points]</span> </div>
# 
# We first use keras incoporated in TensorFlow 2.x for loading the training and testing sets.

# In[1]:


import tensorflow as tf
from tensorflow import keras


# In[2]:


tf.random.set_seed(1234)


# We first use keras datasets in TF 2.x to load Fashion MNIST dataset.

# In[3]:


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full_img, y_train_full), (X_test_img, y_test) = fashion_mnist.load_data()


# The shape of X_train_full_img is $(60000, 28, 28 )$ and that of X_test_img is $(10000, 28, 28)$. We next convert them to matrices of vectors and store in X_train_full and X_test.

# In[4]:


num_train = X_train_full_img.shape[0]
num_test = X_test_img.shape[0]
X_train_full = X_train_full_img.reshape(num_train,-1)
X_test = X_test_img.reshape(num_test, -1)
print(X_train_full.shape, y_train_full.shape)
print(X_test.shape, y_test.shape)


# ####  <span style="color:red">**Question 2.2**</span>. Preprocess the dataset and split into training, validation, and testing datasets
# 
# <div style="text-align: right"> <span style="color:red">[5 points]</span> </div>
# 
# You need to write the code to address the following requirements:
# - Print out the dimensions of X_train_full and X_test
# - Use $10 \%$ of X_train_full for validation and the rest of X_train_full for training. This splits X_train_full and y_train_full into X_train, y_train ($90 \%$) and X_valid, y_valid ($10 \%$).
# - Finally, scale the pixels of X_train, X_valid, and X_test to $[0,1]$) (i.e., $X = X/255.0$).
# 
# You have now the separate training, validation, and testing sets for training your model.
# 
# 

# In[5]:


import math
N = X_train_full.shape[0]
i = math.floor(0.9*N)
X_train, y_train = X_train_full[:i], y_train_full[:i]
X_valid, y_valid = X_train_full[i:], y_train_full[i:]
X_train, X_valid, X_test = X_train/255.0, X_valid/255.0, X_test/255.0


# ####  <span style="color:red">**Question 2.3**</span>. Visualize some images in the training set with labels
# 
# <div style="text-align: right"> <span style="color:red">[5 points]</span> </div>
# 
# You are required to write the code to show **random** $36$ images in X_train_full_img (which is an array of images) with labels as in the following figure. Note that the class names of Fashion MNIST are as follows 
# - "1:T-shirt/top", "2:Trouser", "3:Pullover", "4:Dress", "5:Coat", "6:Sandal", "7:Shirt", "8:Sneaker", "9:Bag", "10:Ankle boot"
# 
# <img src="Figures/Fashion_MNIST.png" width="450" align="center"/>

# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


# YOU ARE REQUIRED TO INSERT YOUR CODES IN THIS CELL
from random import randint
num_row = 6
num_col = 6
images = X_train_full_img
labels = y_train_full
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num_row*num_col):
    index = randint(0,X_train_full_img.shape[0])
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[index], cmap='gray')
    ax.set_title('Label: {}'.format(labels[index]))
plt.show()


# ####  <span style="color:red">**Question 2.4**</span>. Write code for the feed-forward neural net using TF 2.x
# 
# <div style="text-align: right"> <span style="color:red">[5 points]</span> </div>

# We now develop a feed-forward neural network with the architecture $784 \rightarrow 20(ReLU) \rightarrow 40(ReLU) \rightarrow 10(softmax)$. You can choose your own way to implement your network and an optimizer of interest. You should train model in $20$ epochs and evaluate the trained model on the test set.

# In[4]:


#Insert your code here and you can add more cells if necessary
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential


# In[5]:


dnn_model = Sequential()
dnn_model.add(Dense(units=20,  input_shape=(784,), activation='relu'))
dnn_model.add(Dense(units=40, activation='relu'))
dnn_model.add(Dense(units=10, activation='softmax'))
dnn_model.build()


# In[19]:


opt = tf.keras.optimizers.SGD(learning_rate=0.001)
dnn_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[13]:


logdir = "tf_logs/"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
history = dnn_model.fit(x=X_train, y=y_train, batch_size=32, epochs=20, validation_data=(X_valid, y_valid), callbacks=[tensorboard_callback])


# In[14]:


dnn_model.evaluate(X_test, y_test)


# ####  <span style="color:red">**Question 2.5**</span>. Tuning hyper-parameters with grid search
# <div style="text-align: right"> <span style="color:red">[5 points]</span> </div>
# 
# Assume that you need to tune the number of neurons on the first and second hidden layers $n_1 \in \{20, 40\}$, $n_2 \in \{20, 40\}$  and the used activation function  $act \in \{sigmoid, tanh, relu\}$. The network has the architecture pattern $784 \rightarrow n_1 (act) \rightarrow n_2(act) \rightarrow 10(softmax)$ where $n_1, n_2$, and $act$ are in their grides. Write the code to tune the hyper-parameters $n_1, n_2$, and $act$. Note that you can freely choose the optimizer and learning rate of interest for this task.

# In[ ]:


import numpy as np
activations = ['sigmoid','tanh','relu']
list_n1 = [20,40]
list_n2 = [20,40]
best_acc= - np.inf
best_history = None
dnn_model = Sequential()
opt = tf.keras.optimizers.SGD(learning_rate=0.01)

for act in activations:
    for n1 in list_n1:
        for n2 in list_n2:
            dnn_model.add(Dense(units=n1,  input_shape=(784,), activation=act))
            dnn_model.add(Dense(units=n2, activation=act))
            dnn_model.add(Dense(units=10, activation='softmax'))
            dnn_model.build()
            dnn_model.compile(optimizer=opt, 
                              loss='sparse_categorical_crossentropy', 
                              metrics=['accuracy'])
            print("Training with n1 = {}, n2 = {} , activation function = {}".format(n1,n2,act))
            history = dnn_model.fit(x=X_train, y=y_train, batch_size=32, epochs=20, verbose = 0) 
            valid_loss, valid_acc = dnn_model.evaluate(X_test, y_test)
            print('\tvalid acc = {}, valid loss = {}'.format(valid_acc, valid_loss))
            if(valid_acc > best_acc):
                best_acc = valid_acc
                best_model = dnn_model
                best_n1 = n1
                best_n2 = n2
                best_act = act
                best_history = history
print('\nThe best model is with with n1 = {}, n2 = {} , activation function = {}'.format(best_n1,best_n2,best_act))


# ####  <span style="color:red">**Question 2.6**</span>. Experimenting with **the label smoothing** technique
# <div style="text-align: right"> <span style="color:red">[5 points]</span> </div>
# 
# Implement the label smoothing technique (i.e., [link for main paper](https://papers.nips.cc/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf) from Goeff Hinton) by yourself. Note that you cannot use the built-in label-smoothing loss function in TF2.x. Try the label smoothing technique with $\alpha =0.1, 0.15, 0.2$ and report the performances. You need to examine the label smoothing technique with the best architecture obtained in **Question 2.5**.

# In[ ]:


#Insert your code here. You can add more cells if necessary



# ## <span style="color:#0b486b">Part 3: Convolutional Neural Networks and Image Classification</span>
# 
# **<div style="text-align: right"><span style="color:red">[Total marks for this part: 40 points]</span></div>**

# **This part of the asssignment is designed to assess your knowledge and coding skill with Tensorflow as well as hands-on experience with training Convolutional Neural Network (CNN).**

# **The dataset we use for this part is a small animal dataset consisting of $5,000$ images of cats, dogs, fishes, lions, chickens, elephants, butterflies, cows, spiders, and horses, each of which has 500 images. You can download the dataset at [download here](https://drive.google.com/file/d/1bEwEx72lLrjY_Idj_FgV22atIdjtCV66/view?usp=sharing) and then decompress to the folder `datasets\Animals` in your assignment folder.**
# 
# **Your task is to build a CNN model using *TF 2.x* to classify these animals. You're provided with the module <span style="color:red">models.py</span>, which you can find in the assignment folder, with some of the following classes:**

# 1. `AnimalsDatasetManager`: Support with loading and spliting the dataset into the train-val-test sets. It also supports generating next batches for training. `AnimalsDatasetManager` will be passed to CNN model for training and testing.
# 2. `DefaultModel`: A base class for the CNN model.
# 3. `YourModel`: The class you'll need to implement for building your CNN model. It inherits some useful attributes and functions from the base class `DefaultModel`

# Firstly, we need to run the following cells to load and preprocess the Animal dataset.

# In[15]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# Install the package `imutils` if you have not installed yet

# In[16]:


get_ipython().system(' pip install imutils')


# In[22]:


import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
import models
from models import SimplePreprocessor, AnimalsDatasetManager, DefaultModel


# In[56]:


def create_label_folder_dict(adir):
    sub_folders= [folder for folder in os.listdir(adir)
                  if os.path.isdir(os.path.join(adir, folder))]
    label_folder_dict= dict()
    for folder in sub_folders:
        item= {folder: os.path.abspath(os.path.join(adir, folder))}
        label_folder_dict.update(item)
    return label_folder_dict


# In[57]:


label_folder_dict= create_label_folder_dict("./datasets/Animals")


# The below code helps to create a data manager that contains all relevant methods used to manage and process the experimental data. 

# In[58]:


sp = SimplePreprocessor(width=32, height=32)
data_manager = AnimalsDatasetManager([sp])
data_manager.load(label_folder_dict, verbose=100)
data_manager.process_data_label()
data_manager.train_valid_test_split()


# Note that the object `data_manager` has the attributes relating to *the training, validation, and testing sets* as shown belows. You can use them in training your developped models in the sequel.

# In[59]:


print(data_manager.X_train.shape, data_manager.y_train.shape)
print(data_manager.X_valid.shape, data_manager.y_valid.shape)
print(data_manager.X_test.shape, data_manager.y_test.shape)
print(data_manager.classes)


# We now run the **default model** built in the **models.py** file which serves as a basic baseline to start the investigation. Follow the following steps to realize how to run a model and know the built-in methods associated to a model developped in the DefaultModel class.

# We first initialize a default model from the DefaultModel class. Basically, we can define the relevant parameters of training a model including `num_classes`, `optimizer`, `learning_rate`, `batch_size`, and `num_epochs`.

# In[60]:


network1 = DefaultModel(name='network1',
                       num_classes=len(data_manager.classes),
                       optimizer='sgd',
                       batch_size= 128,
                       num_epochs = 20,
                       learning_rate=0.5)


# The method `build_cnn()` assists us in building your convolutional neural network. You can view the code (in the **models.py** file) of the model behind a default model to realize how simple it is. Additionally, the method `summary()` shows the architecture of a model.

# In[61]:


network1.build_cnn()
network1.summary()


# To train a model regarding to the datasets stored in `data_manager`, you can invoke the method `fit()` for which you can specify the batch size and number of epochs for your training. 

# In[62]:


network1.fit(data_manager, batch_size = 64, num_epochs = 20)


# Here you can compute the accuracy of your trained model with respect to a separate testing set.

# In[63]:


network1.compute_accuracy(data_manager.X_test, data_manager.y_test)


# Below shows how you can inspect the training progress.

# In[64]:


network1.plot_progress()


# You can use the method `predict()` to predict labels for data examples in a test set.

# In[65]:


network1.predict(data_manager.X_test[0:10])


# Finally, the method `plot_prediction()` visualizes the predictions for a test set in which several images are chosen to show the predictions.

# In[66]:


network1.plot_prediction(data_manager.X_test, data_manager.y_test, data_manager.classes)


# <span style="color:red">**Question 3.1**</span> **After running the above cells to train the default model and observe the learning curve. Report your observation (i.e. did the model learn well? if not, what is the problem? What would you do to improve it?). Write your answer below.**
# 
# <div style="text-align: right"> <span style="color:red">[4 points]</span> </div>

# *#Your answer and observation here*
# 
# The model does not actually learn very well. , the possible problem is that the number of convolutional layers is insufficient. One possible way to improve the model is to increase the number of convolutional layers and neurons per layer and reduce the bias to improve the performance of the model in terms of accuracy and loss.
# 
# 

# **For questions 3.2 to 3.9, you'll need to write your own model in a way that makes it easy for you to experiment with different architectures and parameters. The goal is to be able to pass the parameters to initialize a new instance of `YourModel` to build different network architectures with different parameters. Below are descriptions of some parameters for `YourModel`, which you can find in function `__init__()` for the class `DefaultModel`:**

# 1. `num_blocks`: an integer specifying the number of blocks in our network. Each block has the pattern `[conv, batch norm, activation, conv, batch norm, activation, mean pool, dropout]`. All convolutional layers have filter size $(3, 3)$, strides $(1, 1)$ and 'SAME' padding, and all mean pool layers have strides $(2, 2)$ and 'SAME' padding. The network will consists of a few blocks before applying a linear layer to output the logits for the softmax layer.
# 
# 2. `feature_maps`: the number of feature maps in the first block of the network. The number of feature_maps will double in each of the following block. To make it convenient for you, we already calculated the number of feature maps for each block for you in line $106$
# 3. `drop_rate`: the keep probability for dropout. Setting `drop_rate` to $0.0$ means not using dropout. 
# 4. `batch_norm`: the batch normalization function is used or not. Setting `batch_norm` to `None` means not using batch normalization. 
# 5. The `skip connection` is added to the output of the second `batch norm`. Additionally, your class has a boolean property (i.e., instance variable) named `use_skip`. If `use_skip=True`, the skip connectnion is enable. Otherwise, if `use_skip=False`, the skip connectnion is disable.
# 
# Below is the architecture of one block:
# 
# <img src="Figures/OneBlock.png" width="350" align="center"/>
# 
# Below is the architecture of the entire deep net with `two blocks`:
# 
# <img src="Figures/NetworkArchitecture.png" width="1200" align="center"/>

# Here we assume that the first block has `feature_maps = feature_maps[0] = 32`. Note that the initial number of feature maps of the first block is declared in the instance variable `feature_maps` and is multiplied by $2$ in each follpwing block. 

# In[67]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


# In[68]:


tf.random.set_seed(1234)


# <span style="color:red">**Question 3.2**</span> **Write the code of the `YourModel` class here. Note that this class will inherit from the `DefaultModel` class. You'll only need to re-write the code for the `build_cnn` method in the `YourModel` class from the cell below. Note that the `YourModel` class   is inherited from the `DefaultModel` class.**
# 
# <div style="text-align: right"> <span style="color:red">[4 points]</span> </div>

# In[69]:


class YourModel(DefaultModel):
    def __init__(self,
                 name='network1',
                 width=32, height=32, depth=3,
                 num_blocks=2,
                 feature_maps=32,
                 num_classes=4, 
                 drop_rate=0.2,
                 batch_norm = None,
                 is_augmentation = False,
                 activation_func='relu',
                 use_skip = True,
                 optimizer='adam',
                 batch_size=10,
                 num_epochs= 20,
                 learning_rate=0.0001,
                 verbose= True):
        super(YourModel, self).__init__(name, width, height, depth, num_blocks, feature_maps, num_classes, drop_rate, batch_norm, is_augmentation, 
                                        activation_func, use_skip, optimizer, batch_size, num_epochs, learning_rate, verbose)
    
    def build_cnn(self):
        #Insert your code here
        self.model = models.Sequential()
        if self.use_skip:
            self.model.add(layers.Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)))
        else:
            self.model.add(layers.Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Conv2D(32, (3,3), padding='same'))
            self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(layers.Dropout(rate=self.drop_rate))
        if self.use_skip:
            self.model.add(layers.Conv2D(64, (3,3), padding='same'))
        else :
            self.model.add(layers.Conv2D(64, (3,3), padding='same'))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Conv2D(64, (3,3), padding='same'))
            self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(layers.Dropout(rate=self.drop_rate))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.num_classes, activation='softmax'))
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    


# <span style="color:red">**Question 3.3**</span> **Once writing your own model, you need to compare two cases: (i) *using the skip connection* and (ii) *not using the skip connection*. You should set the instance variable `use_skip` to either `True` or `False`. For your runs, report which case is better and if you confront overfitting in training.**
#     
# <div style="text-align: right"> <span style="color:red">[6 points]</span> </div>

# #*Write your report and observation here*
# 
# The performance of the model with use_skip = False is better than that of the model with use_skip = True.
# The model has multiple convolutional layers and a batch normalization process. Therefore, the chance of overfitting is very high. By enabling skip connection, the model can skip some convolutional layers and normalization processes, thus preventing the model from being overtrained by the training data.
# When we observe the accuracy of both models, in this case the convolutional layers and the batch normalization process do not lead to the problem of overtraining, while they can be considered suitable for the model to train the existing training dataset. Therefore, based on the current training dataset, skipping connections may not be necessary.
# 
# 
# 

# In[70]:


our_network_skip = YourModel(name='network1',
                     feature_maps=32,
                     num_classes=len(data_manager.classes),
                     num_blocks=3,
                     drop_rate= 0.0, 
                     batch_norm=True, 
                     use_skip = True,
                     optimizer='adam',
                     learning_rate= 0.001)
our_network_skip.build_cnn()
our_network_skip.summary()


# In[71]:


our_network_skip.fit(data_manager, batch_size=32, num_epochs=20)


# In[72]:


our_network_skip.compute_accuracy(data_manager.X_test, data_manager.y_test)


# In[73]:


our_network_no_skip = YourModel(name='network1',
                     feature_maps=32,
                     num_classes=len(data_manager.classes),
                     num_blocks=3,
                     drop_rate= 0.0, 
                     batch_norm=True, 
                     use_skip = False,
                     optimizer='adam',
                     learning_rate= 0.001)
our_network_no_skip.build_cnn()
our_network_no_skip.summary()


# In[74]:


our_network_no_skip.fit(data_manager, batch_size=32, num_epochs=20)


# In[75]:


our_network_no_skip.compute_accuracy(data_manager.X_test, data_manager.y_test)


# <span style="color:red">**Question 3.4**</span> **Now, let us tune the $num\_blocks \in \{2,3,4\}$, $use\_skip \in \{True, False\}$, and $learning\_rate \in \{0.001, 0.0001\}$. Write your code for this tuning and report the result of the best model on the testing set. Note that you need to show your code for tuning and evaluating on the test set to earn the full marks. During tuning, you can set the instance variable `verbose` of your model to `False` for not showing the training details of each epoch.**
#  
# <div style="text-align: right"> <span style="color:red">[4 points]</span> </div>

# #*Report the best parameters and the testing accuracy here*
# 
# .....

# In[3]:


#Insert your code here. You can add more cells if necessary


# <span style="color:red">**Question 3.5**</span> **We now try to apply data augmentation to improve the performance. Extend the code of the class `YourModel` so that if the attribute `is_augmentation` is set to `True`, we apply the data augmentation. Also you need to incorporate early stopping to your training process. Specifically, you early stop the training if the valid accuracy cannot increase in three consecutive epochs.**
#    
# <div style="text-align: right"> <span style="color:red">[4 points]</span> </div>

# In[77]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Wtire your code in the cell below. Hint that you can rewrite the code of the `fit` method to apply the data augmentation. In addition, you can copy the code of `build_cnn` method above to reuse here.

# In[ ]:


class YourModel(DefaultModel):
    def __init__(self,
                 name='network1',
                 width=32, height=32, depth=3,
                 num_blocks=2,
                 feature_maps=32,
                 num_classes=4, 
                 drop_rate=0.2,
                 batch_norm = None,
                 is_augmentation = False,
                 activation_func='relu',
                 use_skip = True,
                 optimizer='adam',
                 batch_size=10,
                 num_epochs= 20,
                 learning_rate=0.0001):
        super(YourModel, self).__init__(name, width, height, depth, num_blocks, feature_maps, num_classes, drop_rate, batch_norm, is_augmentation, 
                                        activation_func, use_skip, optimizer, batch_size, num_epochs, learning_rate)
    
    def build_cnn(self):
        #reuse code of previous section here

    
    
    def fit(self, data_manager, batch_size=None, num_epochs=None, is_augmentation = False):
        #insert your code here

        


# <span style="color:red">**Question 3.6**</span> **Leverage your best model with the data augmentation and try to observe the difference in performance between using data augmentation and non-using it.**
#    
# <div style="text-align: right"> <span style="color:red">[4 points]</span> </div>

# #*Write your answer and observation here*
# 
# .....

# In[2]:


#Insert your code here. You can add more cells if necessary


# <span style="color:red">**Question 3.7**</span> **Exploring Data Mixup Technique for Improving Generalization Ability.**
#    
# <div style="text-align: right"> <span style="color:red">[4 points]</span> </div>
# 
# Data mixup is another super-simple technique used to boost the generalization ability of deep learning models. You need to incoroporate data mixup technique to the above deep learning model and experiment its performance. There are some papers and documents for data mixup as follows:
# - Main paper for data mixup [link for main paper](https://openreview.net/pdf?id=r1Ddp1-Rb) and a good article [article link](https://www.inference.vc/mixup-data-dependent-data-augmentation/).
# 
# You need to extend your model developed above, train a model using data mixup, and write your observations and comments about the result.

# #*Write your answer and observation here*
# 
# .....

# In[1]:


#Insert your code here. You can add more cells if necessary


# <span style="color:red">**Question 3.8**</span> **Attack your best obtained model with PGD, MIM, and FGSM attacks with $\epsilon= 0.0313, k=20, \eta= 0.002$ on the testing set. Write the code for the attacks and report the robust accuracies. Also choose a random set of 20 clean images in the testing set and visualize the original and attacked images.**
#    
# <div style="text-align: right"> <span style="color:red">[5 points]</span> </div>

# In[ ]:


#Insert your code here. You can add more cells if necessary


# <span style="color:red">**Question 3.9**</span> **Train a robust model using adversarial training with PGD ${\epsilon= 0.0313, k=10, \eta= 0.002}$. Write the code for the adversarial training and report the robust accuracies. After finishing the training, you need to store your best robust model in the folder `./models` and load the model to evaluate the robust accuracies for PGD, MIM, and FGSM attacks with $\epsilon= 0.0313, k=20, \eta= 0.002$ on the testing set.**
#    
# <div style="text-align: right"> <span style="color:red">[5 points]</span> </div>

# In[ ]:


#Insert your code here. You can add more cells if necessary


# The following is an exploring question with bonus points. It is great if you try to do this question, but it is **totally optional**. In this question, we will investigate a recent SOTA technique to improve the generalization ability of deep nets named *Sharpness-Aware Minimization (SAM)* ([link to the main paper](https://openreview.net/pdf?id=6Tm1mposlrM)).  Furthermore, SAM is simple and efficient technique, but roughly doubles the training time due to its required computation. If you have an idea to improve SAM, it would be a great paper to top-tier venues in machine learning and computer vision. Highly recommend to give it a try. 

# <span style="color:red">**Question 3.10**</span> (**additionally exploring question**) Read the SAM paper ([link to the main paper](https://openreview.net/pdf?id=6Tm1mposlrM)). Try to apply this techique to the best obtained model and report the results. For the purpose of implementating SAM, we can flexibly add more cells and extensions to the `model.py` file.
# 
# <div style="text-align: right"> <span style="color:red">[5 points]</span> </div>

# In[125]:


#Insert your code here. You can add more cells if necessary



# --- 
# **<div style="text-align: center"> <span style="color:black">END OF ASSIGNMENT</span> </div>**
# **<div style="text-align: center"> <span style="color:black">GOOD LUCK WITH YOUR ASSIGNMENT 1!</span> </div>**
