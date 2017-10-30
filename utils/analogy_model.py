
import itertools
import numpy as np
import gzip
import os
import sklearn.model_selection
import pickle 
import tensorflow as tf

class DataBatcher:
    
    def __init__(self, X, y,  embeds, batch_size, epochs=1):
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.embeds = embeds
        self.epoch = epochs
    
    def _get_inp_vector(self, root, other):
        embeds = self.embeds
        w1, w2 = root.split("\t")
        w3, w4 = other.split("\t")
        try:
            v1 = embeds[w1.lower().strip()]
            v2 = embeds[w2.lower().strip()]
            v3 = embeds[w3.lower().strip()]
            v4 = embeds[w4.lower().strip()]
            inp = []
            inp.extend(v1)
            inp.extend(v2)
            inp.extend(v3)
            inp.extend(v4)
            return inp
        except:
            return None
        
    def data_instances(self, X, y):
        n = len(X)
        for _ in range(self.epoch):
            perm = np.random.permuation(len(X))
            X, y = X[perm], y[perm]
            for i in range(n):
                xi = X[i]
                yi = y[i]
                pos_sample = np.random.choice(X[y == yi])
                neg_sample = np.random.choice(X[y != yi])
                yield X[i], pos_sample, neg_sample
        
    def next_batch(self):
        epochDone = 0
        
        batch = []
        for instance in self.data_instances(X, y, epoch=self.epoch):
            root, pos, neg = instance
            pos_inp = self._get_inp_vector(root, pos)
            neg_inp = self._get_inp_vector(root, neg)
            if(pos_inp is None or neg_inp is None):
                continue
            batch.append(pos_inp)
            batch.append(neg_inp)
            if(len(batch) == batch_size):
                yield batch
                print("Batch returned")
                batch = []

class AnalogyModel:
    
    def __init__(self, input_dim, hidden_layers):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
    
    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name)

    def hinge_loss(self, scores):
        pairs = tf.reshape(scores, (-1,2))
        margin = 1 + pairs[:,1] - pairs[:,0]
        loss = tf.reduce_mean(tf.nn.relu(margin))
        return loss
        
    def _add_placeholders(self):
        self.x = tf.placeholder("float", [None, self.input_dim], name="x")
        

    def _add_hidden_layers(self):
        h = self.x
        last_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_layers):
            W = self.weight_variable([last_dim, hidden_dim], name="w"+str(i))
            b = self.bias_variable([hidden_dim], name="b"+str(i))
            h = tf.matmul(h, W) + b
            last_dim = hidden_dim
        U = self.weight_variable([last_dim,1], name="U")
        self.score = tf.matmul(h, U)
        
        
    def _add_train_op(self):
        self._loss = self.hinge_loss(self.score)
        tf.scalar_summary('loss', self._loss)
        optimizer = tf.train.AdagradOptimizer(0.2)
        self._train_op = optimizer.minimize(self._loss)
        
    def build_graph(self, mode):
        self._add_placeholders()
        self._add_hidden_layers()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if(mode == "train"):
            self._add_train_op()
            
        self._summaries = tf.merge_all_summaries()
        
        
    def run_train_step(self, sess, batch_x):
        feed = {self.x: batch_x}
        to_return = [self._train_op, self._summaries, self._loss, self.global_step]
        return sess.run(to_return, feed_dict=feed)
    
    def run_test_step(self, sess, batch_x):
        feed = {self.x: batch_x}
        to_return = [self.score, self.global_step]
        return sess.run(to_return, feed_dict=feed)



def generate_folds(wholeDict, n_splits=5):
    X = []
    y = []
    num = 1
    for k, v in wholeDict.items():
        X.extend(v)
        y.extend([num]*len(v))
        num += 1
    X = np.asarray(X)
    y = np.asarray(y)
    
    skf = sklearn.model_selection.StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in skf.split(X, y):
        yield X[train_index], y[train_index], X[test_index], y[test_index]
        
        
def load_glove():
    vect = "Q1/glove.6B.300d.txt.gz"
    vectorFile = gzip.open(vect, 'r')
    embeds = {}
    for line in vectorFile:
        comps = line.split()
        embeds[comps[0].strip().decode('utf-8')] = comps[1:]
        if(len(comps[1:]) != 300):
            print (comps[0])
    return embeds

log_root = "Q1/model/logs"
train_dir = "Q1/model/logs/train"
    

def get_req_embeds(data_dict, embeds):
    gmodel = {}
    emodelfile = "Q1/req_embeds.pkl"
    for k, v in data_dict.items():
        for ins in v:
            words = ins.split("\t")
            try:
                gmodel[words[0]] = embeds[words[0]]
            except:
                pass
            try:
                gmodel[words[1]] = embeds[words[1]]
            except:
                pass
    pickle.dump(gmodel, open(emodelfile, "wb"))


def load_embeds():
    emodelfile = "Q1/req_embeds.pkl"
    return pickle.load(open(emodelfile, "rb"))
    
    
def load_trainDict():
    # Dictionary of training pairs for the analogy task
    trainDict = dict()
    analogyTrainPath = "Q1/wordRep/"
    for subDirs in os.listdir(analogyTrainPath):
        for files in os.listdir(analogyTrainPath+subDirs+'/'):
            f = open(analogyTrainPath+subDirs+'/'+files).read().splitlines()
            trainDict[files] = f
    return trainDict
    
def _running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.999):
    """Calculate the running average of losses."""
    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
        running_avg_loss = min(running_avg_loss, 12)
        loss_sum = tf.Summary()
        loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
        summary_writer.add_summary(loss_sum, step)
        sys.stdout.write('running_avg_loss: %f\n' % running_avg_loss)
    return running_avg_loss
    
def _train(model, data_batcher):
    """Runs model training."""
  
    model.build_graph("train")
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=0.25)
    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    summary_writer = tf.train.SummaryWriter(train_dir)
    sv = tf.train.Supervisor(logdir=log_root,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=300,
                             global_step=model.global_step)

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = sv.prepare_or_wait_for_session(config=config)
    running_avg_loss = 0
    step = 0
    done = False
    # sess.run(tf.initialize_all_variables())
    while not sv.should_stop() and not done:
        try:
            x_batch = data_batcher.next_batch()
            (_, summaries, loss, train_step) = model.run_train_step(sess, x_batch)
            summary_writer.add_summary(summaries, train_step)
            running_avg_loss = _running_avg_loss(running_avg_loss, loss, summary_writer, train_step)
            step += 1
            if step % 100 == 0:
                summary_writer.flush()
        except:
            done = True
            summary_writer.flush()
        
    sv.Stop()
    return running_avg_loss
        
        
def _test(model, data_batcher, n_batches):
    """Runs model training."""
  
    model.build_graph()
    saver = tf.train.Saver()
    
    ckpt_state = tf.train.get_checkpoint_state(log_root)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        tf.logging.info('No model to decode yet at %s', log_root)
        return False

    tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
    ckpt_path = os.path.join(
        log_root, os.path.basename(ckpt_state.model_checkpoint_path))
    tf.logging.info('renamed checkpoint path %s', ckpt_path)
    saver.restore(sess, ckpt_path)
    
    n_done = 0
    
    if(n_batches is None):
        cond = lambda x: True
    else:
        cond = lambda x: x < n_batches
    scores_all = []
    while cond(n_done):
        try:
            x_batch = data_batcher.next_batch()
            (scores, train_step) = model.run_test_step(sess, x_batch)
            scores_all.extend(scores)
            n_done += 1
        except:
            cond = lambda x:False
    return scores_all

def accuracy(scores):
    correct = 0
    for i in range(0, len(scores), 2):
        if(scores[i] > scores[i+1]):
            correct += 1
    return float(correct)*2/len(scores)


def main():
    embeds = load_embeds()
    trainDict = load_trainDict()
    for fold in generate_folds(trainDict):
        model = AnalogyModel(1200, [500,200])
        trainX, trainY, testX, testY = fold
        print(len(trainX)) 
        data_batcher = DataBatcher(trainX, trainY, embeds, batch_size=100, epochs=1)
        _train(model, data_batcher)
#         model = AnalogyModel(1200, [500,200])
#         data_batcher = DataBatcher(testX, testY, embeds, batch_size=100, epochs=1)
#         scores = _test(model, data_batcher, n_batches=10)
#         print(accuracy(scores))
        break
        
        
main()