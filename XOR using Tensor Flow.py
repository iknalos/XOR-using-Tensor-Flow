import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 

n_inputs = 2  
n_outputs = 1
n_hidden = 4

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None, n_outputs), name="y")

# Construct model
W1 = tf.Variable(tf.random_uniform([n_inputs,n_hidden], -1, 1), name = "W1")
W2 = tf.Variable(tf.random_uniform([n_hidden,n_outputs], -1, 1), name = "W2")

B1 = tf.Variable(tf.zeros([1,n_hidden]), name="bias1")
B2 = tf.Variable(tf.zeros([1,n_outputs]), name="bias2")


#activation & pred = tf.nn.sigmoid(Z)

Z1 = tf.matmul(X, W1) + b1
pred1 = tf.nn.sigmoid(Z1)

Z2 = tf.matmul(pred1, W2) + b2
pred2 = tf.nn.sigmoid(Z2)

# Minimize error using cross entropy
loss = tf.reduce_mean(tf.reduce_sum(tf.nn.l2_loss(y-pred2)))

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 10000
with tf.Session() as sess:
    X_batch = [[1, 0], [1, 0.01], [1, 0.02], [1, 0.95], [1, 1.01], [1, 0.99], [1, 1], [0, 0], [0.01, 0.1]]
    y_batch = [[1], [1], [1], [1], [1],[1], [1], [0], [0]]
    
    init.run()
    for epoch in range(n_epochs):
        loss_val, _=sess.run([loss, training_op], feed_dict={X: X_batch, y: y_batch})
       # print("loss: ", loss_val)
        print('   W1: ')
        for element in sess.run(W1):
            print('    ',element)
        
        print('   W2: ')
        for element in sess.run(W2):
            print('    ',element)
        
        print('   B1: ')
        for element in sess.run(B1):
            print('    ',element)
        
        print('   B2: ')
        for element in sess.run(B2):
            print('    ',element)

    X_new = [[1, 0.01], [1, 0.9], [1, 1], [1, 0.99], [0, 0.01], [0.01, 0.01]]
    y_pred = pred.eval(feed_dict={X: X_new})
    y_actuals = [1, 1, 1, 1, 0, 0]

    print("Predicted classes:", np.round(y_pred))
    print("Actual classes:   ", y_actuals)
