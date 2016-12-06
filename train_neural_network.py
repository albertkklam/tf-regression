# Create network architecture

n_nodes_hl1 = 400
n_nodes_hl2 = 200
n_nodes_hl3 = 100
n_nodes_hl4 = 50
n_nodes_hl5 = 10

n_epochs = 20

INPUT_DIMENSION = x_train.shape[1]
OUTPUT_DIMENSION = 1
BATCH_SIZE = 10000

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'weights': tf.Variable(tf.random_normal([INPUT_DIMENSION, n_nodes_hl1])), 'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}
hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}
hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}
hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])), 'bias': tf.Variable(tf.random_normal([n_nodes_hl4]))}
hidden_5_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])), 'bias': tf.Variable(tf.random_normal([n_nodes_hl5]))}
output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl5, OUTPUT_DIMENSION])), 'bias': tf.Variable(tf.random_normal([OUTPUT_DIMENSION]))}

# Define model

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)
    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['bias'])
    l4 = tf.nn.relu(l4)
    l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['bias'])
    l5 = tf.nn.sigmoid(l5)
    output = tf.matmul(l5, output_layer['weights']) + output_layer['bias']
    return output
    
def train_neural_network(x):
    
    prediction = neural_network_model(x)
    
    # Create the elastic net cost function
    elastic_param1 = tf.constant(1.)
    elastic_param2 = tf.constant(1.)
    l1_cost = tf.reduce_mean(tf.abs(output_layer['weights']))
    l2_cost = tf.reduce_mean(tf.square(output_layer['weights']))
    e1_term = tf.mul(elastic_param1, l1_cost) + 1e-10
    e2_term = tf.mul(elastic_param2, l2_cost) + 1e-10
    cost = tf.add(tf.add(tf.reduce_mean(tf.square(y - prediction)), e1_term), e2_term)
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        
        sess.run(tf.initialize_all_variables())
        cost_vec = []
        for epoch in range(n_epochs):
        
            epoch_loss = 0
            i = 0
            while i < len(x_train):
            
                start = i
                end = i + BATCH_SIZE
                batch_x = np.array(x_train[start:end])
                batch_y = np.array(y_train[start:end])
                o, temp_cost = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                cost_vec.append(temp_cost)
                epoch_loss += temp_cost
                i += BATCH_SIZE
                
            print('Epoch', epoch+1, 'completed out of', n_epochs, 'loss: {:.3e}'.format(epoch_loss / 1000))
        
        # SMAPE and RMSE are used here, but you can implement your own accuracy measures here
        smape = tf.reduce_mean(tf.div(tf.abs(prediction - y), tf.div(tf.abs(prediction) + tf.abs(y),2)))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(prediction - y)))
        print('Train SMAPE: ', smape.eval({x: x_train, y: y_train}))
        print('Train RMSE: ', rmse.eval({x: x_train, y: y_train}))

        # Plot cost over time
        plt.plot(cost_vec, 'k-')
        plt.title('Cost per epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.show()
        
train_neural_network(x)
