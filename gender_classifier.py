import tensorflow as tf
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
import shutil

# Constants + Hyperparameters
male_feature_path = './feature_data/all_male_feats.npy'
male_labels_path = './feature_data/all_male_labels.npy'
female_feature_path = './feature_data/all_female_feats.npy'
female_labels_path = './feature_data/all_female_labels.npy'

FEATURE_DIM = 4096
RAND_SEED = 231 
epochs = 20
batch_size = 32
lr = 1e-5
drop_p = .25
H1 = 1024
H2 = 128
H3 = 2

def train_model(train, val, test, write_dir):
    """ trains model on train set, writes best model to write_dir and returns training loss curves
    
    :param train, val, test: tupled numpy arrays of (feats, labels)
    :param write_dir: directory to save model to 

    :type: three lists representing training history
    """
    train_set = tf.data.Dataset.from_tensor_slices(train).repeat().batch(batch_size).prefetch(batch_size)

    with tf.name_scope("inputs"):
        x_batch = tf.placeholder(tf.float32, (None, FEATURE_DIM))
        y_batch = tf.placeholder(tf.float32, (None, 2)) # one hots
        is_train = tf.placeholder(tf.bool, name = 'mode') # True for training

    with tf.name_scope("h1"):
        l1 = tf.layers.Dense(units = H1, activation = tf.nn.relu)
        a1 = tf.layers.dropout(l1(x_batch), rate = drop_p, training = is_train)
    with tf.name_scope("h2"):
        l2 = tf.layers.Dense(units = H2, activation = tf.nn.relu)
        a2 =  tf.layers.dropout(l2(a1), rate = drop_p, training = is_train)
    with tf.name_scope("h3"):
        l3 = tf.layers.Dense(units = H3, activation = tf.nn.relu)
        a3 = l3(a2)

    with tf.name_scope("output"):
        probs = tf.nn.softmax(a3)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_batch, logits = a3)
        num_correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(probs, axis = 1), tf.argmax(y_batch, axis = 1)), tf.int32))
        loss = tf.reduce_sum(cross_entropy)
        train_step = train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    sess = tf.Session()
    train_it = train_set.make_initializable_iterator()
    next_el = train_it.get_next()
    sess.run(tf.global_variables_initializer())
    sess.run(train_it.initializer)

    best_val_acc = 0
    epoch_loss_history = []
    batch_loss_history = []
    val_loss_history = []
    steps  = len(train[0]) // batch_size
    feat_val, label_val = val
    print("Init val loss " + str(sess.run(loss, feed_dict = {x_batch : feat_val, y_batch : label_val[:, 0:2],  is_train : False})))

    for epoch in range(epochs):
        epoch_loss =  0.0
        correct_count = 0
        total = 0
        for step in range(steps):
            x_b, y_b = sess.run(next_el)
            y_b_hot = y_b[:, 0:2] # drop age column
            feed_dict = {x_batch : x_b, y_batch : y_b_hot, is_train : True}
            _, batch_loss, scores, correct_preds = sess.run([train_step, loss, probs, num_correct], feed_dict)
            correct_count += correct_preds
            total += batch_size
            epoch_loss += batch_loss
            batch_loss_history.append(batch_loss)
            if step % 50 == 0:
                print("Epoch " + str(epoch)  + " | batch_loss " +  "%.2f" % (batch_loss) + " | batch_acc " + "%.3f" % (correct_preds / batch_size))

        epoch_loss_history.append(epoch_loss / total)
        val_loss, val_correct = sess.run([loss, num_correct], feed_dict = {x_batch : feat_val, y_batch : label_val[:, 0:2],  is_train : False})

        val_loss_history.append(val_loss / len(feat_val))
        print("Epoch " + str(epoch) + " | epoch loss "  + "%.3f" % (epoch_loss) + " | val loss "  + str(val_loss) +
         " | train_acc  " + "%.3f" % (correct_count / total) +  " | val_acc "  +"%.3f" % (val_correct / len(feat_val)))

        if (val_correct / len(feat_val)) > best_val_acc:
            # save model
            best_val_acc = (val_correct / len(feat_val))
            print("New best")
            if os.path.isdir(write_dir + 'best_model/'):
                shutil.rmtree(write_dir + 'best_model/')

            tf.saved_model.simple_save(sess, write_dir + 'best_model/',
                                    inputs = {'x_batch' : x_batch, 'y_batch' : y_batch},
                                    outputs = {'ce' : cross_entropy, 'loss' : loss, 'num_correct' : num_correct})

    return epoch_loss_history, val_loss_history, batch_loss_history


def main():

    # Data prep
    male_feats = np.load(male_feature_path)
    male_labels = np.load(male_labels_path)
    female_feats = np.load(female_feature_path)
    female_labels = np.load(female_labels_path)

    feats = np.concatenate((male_feats, female_feats))
    labels = np.concatenate((male_labels, female_labels))

    np.random.seed(RAND_SEED)
    shuf_inds = np.random.permutation(range(len(feats)))
    feats = feats[shuf_inds]
    labels = labels[shuf_inds]

    feat_train, feat_test, label_train, label_test =  train_test_split(feats, labels, test_size = .3 , shuffle = False)
    feat_val, feat_test, label_val, label_test = train_test_split(feat_test, label_test, test_size = .5, shuffle = False)
    print(sum(feat_train[0, :]))

    print("start train")
    write_dir = './model_' + str(lr) + "_" + str(epochs) + "_" + str(drop_p) + "/" 
    loss_curves = train_model((feat_train, label_train), (feat_val, label_val), (feat_test, label_test), write_dir)

    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)

    pickle.dump(loss_curves, open(write_dir + 'loss_curves.pkl', 'wb'))


if __name__ == '__main__':
    main()