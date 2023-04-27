import streamlit as st
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from intersection import *
import tensorflow as tf
from tensorflow import keras
import keras.backend as K


# set default page config to wide
st.set_page_config(layout="wide")

# Title text of the app
title_txt = "Finding intersection points of curve using Mathematical Methods"
st.markdown(f"<h3 style='text-align: center;'>{title_txt}</h1> <br><br>",unsafe_allow_html=True)

# Next row of app will have 4 columns
#   column-1 : Input for function-1
#   column-2 : Input for function-2
#   column-3 : Input for no. of points to plot
#   column-4 : Display equations of functions
col1,col2,col3,col4 = st.columns([1,2,2,3])
with col1:
    func1 = st.radio("Function-1", ('DoubleCone_1', 'Sphere_1', 'Cylinder_1'))
    func1 = str(func1)
with col2:
    func2 = st.radio("Function-2", ('DoubleCone_2', 'Sphere_2', 'Cylinder_2'))
    func2 = str(func2)
with col3:
    #N = st.slider('Number of points to plot. More the slower', 200, 3000, 500, step=100)
    N=int(st.text_input("Enter the Number of  points to be given as input", 200)) # 1000 points is the default value
with col4:
    st.write(F1_curve(X, func1))
    st.write(F2_curve(X, func2))


# Next row of app will have 2 columns
#   column-1 : Plot for input functions
#   column-2 : Plot for intersection of the functions


col1,col2 = st.columns(2)
x1,y1,z1=[],[],[]
x2,y2,z2=[],[],[]
with col1:
    data1 = generate_F1_sample(N, func1)
    data2 = generate_F2_sample(N, func2)

    x1, y1, z1 = data1[:, 0], data1[:, 1], data1[:, 2]
    trace1 = go.Scatter3d(x=x1, y=y1, z=z1,mode='markers', name="Function_1", marker=dict(size=5,sizemode='diameter'))
    x2, y2, z2 = data2[:, 0], data2[:, 1], data2[:, 2]
    trace2 = go.Scatter3d(x=x2, y=y2, z=z2,mode='markers', name="Function_2", marker=dict(size=5,sizemode='diameter'))

    fig = make_subplots()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=600, width=1000)
    st.plotly_chart(fig, use_container_width=True, height=600, width=1000)

x,y,z=[],[],[] # Define variables to be used in deep Learning
## Plot for Intersection of Curves
data = generate_intersection_sample(N, func1,func2)
data = np.array(data)
x, y, z = data[:, 0], data[:, 1], data[:, 2]

# Start the Deep learning Process :

def return_data_input(data_1,data_2):
    x1, y1, z1 = data_1[:, 0], data_1[:, 1], data_1[:, 2]
    x2, y2, z2 = data_2[:, 0], data_2[:, 1], data_2[:, 2]
    x1y1=x1*y1
    x2y2=x2*y2
    x_2=x1**2+x2**2
    y_2=y1**2+y2**2
    return np.column_stack((x1,y1,x2y2,x2,y2,x1y1))#,x1y1,x2y2

data_train=return_data_input(data1,data2)
#print(data_train.shape)

def custom_loss(y_true,y_pred):
    loss=0
    #print(y_pred.shape)
    #for i in range(y_pred.shape[0]):
    loss+=G(y_pred,func1,func2)
    print(loss)
    #loss=loss**0.5
    return loss

from keras.layers import Layer
import keras.backend as K

class CustomDense(Layer):

  def __init__(self, units=32,activation='None',trainable=True):
      super(CustomDense, self).__init__()
      self.units = units
      self.trainable=trainable
      if activation != 'None':
          self.activation = tf.keras.activations.get(activation)
      else:
        self.activation=activation

  def build(self, input_shape):  
  
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units),dtype='float32'),trainable=self.trainable)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(initial_value=b_init(shape=(self.units,), dtype='float32'),trainable=self.trainable)
    
  def call(self, inputs):  
     
      #if self.activation !=None:
           #return self.activation(tf.matmul(inputs, self.w) + self.b)
      #else:
         #print('Value from custom Layer',G(tf.matmul(inputs, self.w) + self.b,func1,func2))
         #inputs=tf.make_ndarray(inputs)
         return G_2(inputs,func1,func2)

  def compute_output_shape(self, input_shape): 
    return (1,) #(input_shape[0], self.units)

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Dense(32, input_shape=(data_train.shape[1],),kernel_initializer='uniform', activation='relu'),
    keras.layers.Dense(64, activation='linear'), # Hidden layer
    keras.layers.Dense(64, activation='linear'),
    #keras.layers.Dense(64, activation='relu'),
    #keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='linear'),
    keras.layers.Dense(3, activation='linear'),
    CustomDense(1),
    #CustomDense(3,None)
])

# Compile the model with the mean squared error loss and the Adam optimizer
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss='huber_loss')#'huber_loss'

# Train the model using the x and y coordinates as input and the z coordinate as output
model.fit(data_train,z, epochs=200)#np.column_stack((x, y, z))

#model.pop() # Remove Last Layer
print(model)
#model.pop()
new_data1 = generate_F1_sample(N, func1)
new_data2 = generate_F2_sample(N, func2)

data_test=return_data_input(new_data1,new_data2)
plot_data=model.predict(data_test)
print(plot_data)

with col2:  
    trace1 = go.Scatter3d(x=x, y=y, z=z,mode='markers', name="Function_1", marker=dict(size=5,sizemode='diameter'))
    x2, y2, z2 = data2[:, 0], data2[:, 1], data2[:, 2]
    trace2 = go.Scatter3d(x=plot_data[:,0], y=plot_data[:,1], z=plot_data[:,2],mode='markers', name="Function_2", marker=dict(size=5,sizemode='diameter'))

    fig = make_subplots()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=600, width=1000)
    st.plotly_chart(fig, use_container_width=True, height=600, width=1000)


'''
#trace1 = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,mode='markers',name='From Steapest Gradient Descent',marker=dict(size=2,sizemode='diameter'))])
    trace2 = go.Figure(data=[go.Scatter3d(x=plot_data[:, 0], y=plot_data[:, 1], z=plot_data[:, 2],mode='markers',name='From DNN',marker=dict(size=2,sizemode='diameter'))])
    fig = make_subplots()
    #fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=600, width=1000)
    st.plotly_chart(fig, use_container_width=True, height=600, width=1000)
'''




st.write('### Equation of the curve as found from the Deep Learning Network')
col1,col2,col3,col4,col5,col6 = st.columns([3,3,3,3,3,3])
with col1:
    st.write(' ')
    st.write('#### Coefficients')
with col2:
    st.write('##### X')
with col3:
    st.write('##### Y')
with col4:
    st.write('##### XY')
with col5:
    st.write('##### X**2')
with col2:
    st.write('##### Y**2')
    




readmeLink="#### [Mathematical details and approach](https://github.com/vdivakar/curves-intersection-with-gradient-descent/blob/main/README.md)"
st.markdown(readmeLink, unsafe_allow_html=True)

repoLink="#### [Github Repository](https://github.com/vdivakar/curves-intersection-with-gradient-descent)"
st.markdown(repoLink, unsafe_allow_html=True)