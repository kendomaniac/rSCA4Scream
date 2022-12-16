'''
Keras implementation of deep embedder to improve clustering, inspired by:
"Unsupervised Deep Embedding for Clustering Analysis" (Xie et al, ICML 2016)

Definition can accept somewhat custom neural networks. Defaults are from paper.
'''
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tsf
from sklearn.cluster import Birch
import sys
import numpy as np
import keras.backend as K
from keras.initializers import RandomNormal
from keras.layers import Layer, InputSpec
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import SGD
from sklearn.preprocessing import normalize
from keras.callbacks import LearningRateScheduler
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.cm as cm    
from sklearn.manifold import TSNE

if (sys.version[0] == 2):
    import cPickle as pickle
else:
    import pickle
import numpy as np

class ClusteringLayer(Layer):
    '''
    Clustering layer which converts latent space Z of input layer
    into a probability vector for each cluster defined by its centre in
    Z-space. Use Kullback-Leibler divergence as loss, with a probability
    target distribution.
    # Arguments
        output_dim: int > 0. Should be same as number of clusters.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        alpha: parameter in Student's t-distribution. Default is 1.0.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    '''
    def __init__(self, output_dim, input_dim=None, weights=None, alpha=1.0, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.alpha = alpha
        # kmeans cluster centre locations
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = K.variable(self.initial_weights)
        self._trainable_weights = [self.W]

    def call(self, x, mask=None):
        q = 1.0/(1.0 + K.sqrt(K.sum(K.square(K.expand_dims(x, 1) - self.W), axis=2))**2 /self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = K.transpose(K.transpose(q)/K.sum(q, axis=1))
        return q

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepEmbeddingClustering(object):
    def __init__(self,
                 n_clusters,
                 input_dim,
                 Atac,
                 bias,
                 sep,
                 path,
                 bName="NULL",
                 encoded=None,
                 decoded=None,
                 alpha=1.0,
                 pretrained_weights=None,
                 cluster_centres=None,
                 batch_size=256,
                 **kwargs):
        #import pandas as pd
        #import keras
        #import tensorflow as tsf
        super(DeepEmbeddingClustering, self).__init__()

        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.encoded = encoded
        self.decoded = decoded
        self.alpha = alpha
        self.pretrained_weights = pretrained_weights
        self.cluster_centres = cluster_centres
        self.batch_size = batch_size
        self.Atac=Atac
        self.learning_rate = 0.01
        self.iters_lr_update = 20000
        self.lr_change_rate = 0.01
        self.path=path
        if bias == "TF":
            tfName="/home/tf.txt"
            tf=pd.read_csv(tfName,sep="\t",header=0)
            transcriptionFactors=np.unique(tf.iloc[:,0])
            relationMatrix=pd.DataFrame(np.zeros((self.Atac.columns.shape[0],transcriptionFactors.shape[0])))
            relationMatrix.index=self.Atac.columns
            relationMatrix.columns=transcriptionFactors
            for i in self.Atac.columns:
                tfTemp=np.unique(tf.iloc[np.where(tf.iloc[:,1]==i)[0],0])
                relationMatrix.loc[i,tfTemp]=1
        if bias == "cytoBands":
            tfName="/home/hg38.p13_105.csv"
            tf=pd.read_csv(tfName,sep=",",header=0)
            transcriptionFactors=np.unique(tf.iloc[:,1])
            relationMatrix=pd.DataFrame(np.zeros((Atac.columns.shape[0],transcriptionFactors.shape[0])))
            relationMatrix.index=Atac.columns
            relationMatrix.columns=transcriptionFactors
            for i in Atac.columns:
                tfTemp=np.unique(tf.iloc[np.where(tf.iloc[:,0]==i)[0],1])
                relationMatrix.loc[i,tfTemp]=1
        if bias == "kegg":
            tfName="/home/kegg.txt"
            tf=pd.read_csv(tfName,sep="\t",header=0)
            transcriptionFactors=np.unique(tf.iloc[:,1])
            relationMatrix=pd.DataFrame(np.zeros((Atac.columns.shape[0],transcriptionFactors.shape[0])))
            relationMatrix.index=Atac.columns
            relationMatrix.columns=transcriptionFactors
            for i in Atac.columns:
                tfTemp=np.unique(tf.iloc[np.where(tf.iloc[:,0]==i)[0],1])
                relationMatrix.loc[i,tfTemp]=1
        if bias == "reactome":
            tfName="/home/reactome.txt"
            tf=pd.read_csv(tfName,sep="\t",header=0)
            transcriptionFactors=np.unique(tf.iloc[:,1])
            relationMatrix=pd.DataFrame(np.zeros((Atac.columns.shape[0],transcriptionFactors.shape[0])))
            relationMatrix.index=Atac.columns
            relationMatrix.columns=transcriptionFactors
            for i in Atac.columns:
                tfTemp=np.unique(tf.iloc[np.where(tf.iloc[:,0]==i)[0],1])
                relationMatrix.loc[i,tfTemp]=1
        if bias == "mirna":
            tfName="/home/mirna.txt"
            tf=pd.read_csv(tfName,sep="\t",header=0)
            transcriptionFactors=np.unique(tf.iloc[:,0])
            relationMatrix=pd.DataFrame(np.zeros((Atac.columns.shape[0],transcriptionFactors.shape[0])))
            relationMatrix.index=Atac.columns
            relationMatrix.columns=transcriptionFactors
            for i in Atac.columns:
                tfTemp=np.unique(tf.iloc[np.where(tf.iloc[:,3]==i)[0],0])
                relationMatrix.loc[i,tfTemp]=1
        if bias == "kinasi":
            tfName="/home/kinase-specific_phosphorylation_sites.csv"
            tf=pd.read_csv(tfName,sep=",",header=0)
            transcriptionFactors=np.unique(tf.iloc[:,0])
            relationMatrix=pd.DataFrame(np.zeros((Atac.columns.shape[0],transcriptionFactors.shape[0])))
            relationMatrix.index=Atac.columns
            relationMatrix.columns=transcriptionFactors
            for i in Atac.columns:
                tfTemp=np.unique(tf.iloc[np.where(tf.iloc[:,1]==i)[0],0])
                relationMatrix.loc[i,tfTemp]=1
        if bias == "immunoSignature":
            tfName="/home/cancer_related_immuno_signatures.csv"
            tf=pd.read_csv(tfName,sep=",",header=0)
            transcriptionFactors=np.unique(tf.iloc[:,0])
            relationMatrix=pd.DataFrame(np.zeros((Atac.columns.shape[0],transcriptionFactors.shape[0])))
            relationMatrix.index=Atac.columns
            relationMatrix.columns=transcriptionFactors
            for i in Atac.columns:
                tfTemp=np.unique(tf.iloc[np.where(tf.iloc[:,1]==i)[0],0])
                relationMatrix.loc[i,tfTemp]=1
        if bias == "ALL":
            tfName="/home/mirnaTFISKinase.csv"
            tf=pd.read_csv(tfName,sep=",",header=0)
            transcriptionFactors=np.unique(tf.iloc[:,0])
            relationMatrix=pd.DataFrame(np.zeros((Atac.columns.shape[0],transcriptionFactors.shape[0])))
            relationMatrix.index=Atac.columns
            relationMatrix.columns=transcriptionFactors
            for i in Atac.columns:
                tfTemp=np.unique(tf.iloc[np.where(tf.iloc[:,1]==i)[0],0])
                relationMatrix.loc[i,tfTemp]=1
        if bias == "CUSTOM":
            tfName="/scratch/"+bName
            tf=pd.read_csv(tfName,sep=sep,header=0)
            transcriptionFactors=np.unique(tf.iloc[:,0])
            relationMatrix=pd.DataFrame(np.zeros((Atac.columns.shape[0],transcriptionFactors.shape[0])))
            relationMatrix.index=Atac.columns
            relationMatrix.columns=transcriptionFactors
            for i in Atac.columns:
                tfTemp=np.unique(tf.iloc[np.where(tf.iloc[:,1]==i)[0],0])
                relationMatrix.loc[i,tfTemp]=1
        relationMatrix=relationMatrix.loc[:, (relationMatrix != 0).any(axis=0)]
        self.transcriptionFactors=np.asarray(relationMatrix.T.index)
        class provaEncoder(keras.constraints.Constraint):
            def __call__(self, w):
                return tsf.math.multiply(w,tsf.constant(np.asarray(relationMatrix),tsf.float32))
        class provaDecoder(keras.constraints.Constraint):
            def __call__(self, w):
                return tsf.math.multiply(w,tsf.transpose(tsf.constant(np.asarray(relationMatrix),tsf.float32)))
        # greedy layer-wise training before end-to-end training:

        self.encoders_dims = [self.input_dim,relationMatrix.shape[1]]

        self.input_layer = Input(shape=(self.input_dim,), name='input')
        init_stddev = 0.01

        self.layer_wise_autoencoders = []
        self.encoders = []
        self.decoders = []
        for i  in range(1, len(self.encoders_dims)):
            
            encoder_activation = 'relu'
            encoder = Dense(self.encoders_dims[i], activation=encoder_activation,input_shape=(self.encoders_dims[i-1],),name='encoder_dense_%d'%i,kernel_constraint=provaEncoder())
            self.encoders.append(encoder)

            decoder_index = len(self.encoders_dims) - i
            decoder_activation = 'relu'
            decoder = Dense(self.encoders_dims[i-1], activation=decoder_activation,name='decoder_dense_%d'%decoder_index,kernel_constraint=provaDecoder())
            self.decoders.append(decoder)

            autoencoder = Sequential([
            #    Dropout(dropout_fraction, input_shape=(self.encoders_dims[i-1],), 
            #            name='encoder_dropout_%d'%i),
                encoder,
            #    Dropout(dropout_fraction, name='decoder_dropout_%d'%decoder_index),
                decoder
            ])
            autoencoder.compile(loss='mse', optimizer=keras.optimizers.Adam( lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
            self.layer_wise_autoencoders.append(autoencoder)

        # build the end-to-end autoencoder for finetuning
        # Note that at this point dropout is discarded
        self.encoder = Sequential(self.encoders)
        self.encoder.compile(loss='mse',optimizer=keras.optimizers.Adam(lr=self.learning_rate,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
        self.decoders.reverse()
        self.autoencoder = Sequential(self.encoders + self.decoders)
        self.autoencoder.compile(loss='mse', optimizer=keras.optimizers.Adam( lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))

        if cluster_centres is not None:
            assert cluster_centres.shape[0] == self.n_clusters
            assert cluster_centres.shape[1] == self.encoder.layers[-1].output_dim

        if self.pretrained_weights is not None:
            self.autoencoder.load_weights(self.pretrained_weights)

    def p_mat(self, q):
        weight = q**2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def initialize(self, X, save_autoencoder=False, layerwise_pretrain_iters=50000, finetune_iters=100000):
        if self.pretrained_weights is None:

            iters_per_epoch = int(len(X) / self.batch_size)
            layerwise_epochs = max(int(layerwise_pretrain_iters / iters_per_epoch), 1)
            finetune_epochs = max(int(finetune_iters / iters_per_epoch), 1)

            print('layerwise pretrain')
            current_input = X
            lr_epoch_update = max(1, self.iters_lr_update / float(iters_per_epoch))
            
            def step_decay(epoch):
                initial_rate = self.learning_rate
                factor = int(epoch / lr_epoch_update)
                lr = initial_rate / (10 ** factor)
                return lr
            lr_schedule = LearningRateScheduler(step_decay)

            for i, autoencoder in enumerate(self.layer_wise_autoencoders):
                print(len(self.layer_wise_autoencoders))
                autoencoder.summary()
                autoencoder.fit(current_input, current_input, 
                                batch_size=self.batch_size, epochs=layerwise_epochs, callbacks=[lr_schedule])
                self.autoencoder.layers[i].set_weights(autoencoder.layers[0].get_weights())
                self.autoencoder.layers[len(self.autoencoder.layers) - i - 1].set_weights(autoencoder.layers[-1].get_weights())
            
            print('Finetuning autoencoder')
            
            #update encoder and decoder weights:
            self.autoencoder.fit(X, X, batch_size=self.batch_size, epochs=finetune_epochs, callbacks=[lr_schedule])

            if save_autoencoder:
                self.autoencoder.save_weights('autoencoder.h5')
        else:
            print('Loading pretrained weights for autoencoder.')
            self.autoencoder.load_weights(self.pretrained_weights)

        # update encoder, decoder
        # TODO: is this needed? Might be redundant...
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].set_weights(self.autoencoder.layers[i].get_weights())

        # initialize cluster centres using k-means
        print('Initializing cluster centres with k-means.')
        if self.cluster_centres is None:
            #brc = Birch(n_clusters=None)
            U = self.encoder.predict(X)
            #brc.fit(U)
            #brc.predict(U)
            #self.n_cluster=len(np.unique(brc.labels_))
            #print("Yojohn")
            #print(self.n_cluster)
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            self.y_pred = kmeans.fit_predict(U)
            self.cluster_centres = kmeans.cluster_centers_

        # prepare DEC model
        #self.DEC = Model(inputs=self.input_layer,
        #                 outputs=ClusteringLayer(self.n_clusters,
        #                                        weights=self.cluster_centres,
        #                                        name='clustering')(self.encoder))
        self.DEC = Sequential([self.encoder,
                             ClusteringLayer(self.n_clusters,
                                                weights=self.cluster_centres,
                                                name='clustering')])
        self.DEC.compile(loss='kullback_leibler_divergence', optimizer='adadelta')
        return

    def cluster_acc(self, y_true, y_pred):
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max())+1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind])*1.0/y_pred.size, w

    def cluster(self, X, y=None,
                tol=0.01, update_interval=None,
                iter_max=1e6,
                #iter_max=1000,
                save_interval=None,
                **kwargs):

        if update_interval is None:
            # 1 epochs
            update_interval = X.shape[0]/self.batch_size
        print('Update interval', update_interval)

        if save_interval is None:
            # 50 epochs
            save_interval = X.shape[0]/self.batch_size*50
        print('Save interval', save_interval)

        assert save_interval >= update_interval

        train = True
        iteration, index = 0, 0
        self.accuracy = []
        contator=0
        while train:
            if contator==100:
                ccc=cm.rainbow(np.linspace(0,1,self.n_clusters))
                print('Converged!!!')
                z = self.encoder.predict(X)
                X_tsne = TSNE(learning_rate=200).fit_transform(z)
                ee=pd.DataFrame(z)
                pca = PCA(n_components=2).fit(z)
                z_2d = pca.transform(z)
                z2=pd.DataFrame(z)
                z2.index=X.index
                z2=z2.T
                z2.index=self.transcriptionFactors
                z2.to_csv(self.path+"/latentSpace.csv")
                clust_2d = pca.transform(self.cluster_centres)
                # save states for visualization
                clustering_output=pd.DataFrame(X.index,columns=["CellName"])
                clustering_output.insert(1, "Belonging_cluster",y_pred)
                clustering_output.insert(2,"ChoordinatesX",X_tsne[:,0])
                clustering_output.insert(3,"ChoordinatesY",X_tsne[:,1])
                clustering_output.to_csv(self.path+"/clustering.output.csv",index=False,sep=",")
                for i in range(self.n_clusters):
                    temp=np.where(y_pred==i)
                    plt.scatter(X_tsne[temp, 0], X_tsne[temp, 1],c=[list(ccc[i])],s=6)
                plt.savefig(self.path+"/clusteringKMEANS.png")
                plt.clf()
                # save DEC model checkpoints
                self.DEC.save(self.path+'/DEC_model_'+'.h5')
                return self.y_pred
            sys.stdout.write('\r')
            # cutoff iteration
            if iter_max < iteration:
                ccc=cm.rainbow(np.linspace(0,1,self.n_clusters))
                print('Reached maximum iteration limit. Stopping training.')
                z = self.encoder.predict(X)
                X_tsne = TSNE(learning_rate=200).fit_transform(z)
                ee=pd.DataFrame(z)
                pca = PCA(n_components=2).fit(z)
                z_2d = pca.transform(z)
                z2=pd.DataFrame(z)
                z2.index=X.index
                z2=z2.T
                z2.index=self.transcriptionFactors
                z2.to_csv(self.path+"/latentSpace.csv")
                clust_2d = pca.transform(self.cluster_centres)
                # save states for visualization
                clustering_output=pd.DataFrame(X.index,columns=["CellName"])
                clustering_output.insert(1, "Belonging_cluster",y_pred)
                clustering_output.insert(2,"ChoordinatesX",X_tsne[:,0])
                clustering_output.insert(3,"ChoordinatesY",X_tsne[:,1])
                clustering_output.to_csv(self.path+"/clustering.output.csv",index=False,sep=",")
                for i in range(self.n_clusters):
                    temp=np.where(y_pred==i)
                    plt.scatter(X_tsne[temp, 0], X_tsne[temp, 1],c=[list(ccc[i])],s=6)
                plt.savefig(self.path+"/clusteringKMEANS.png")
                plt.clf()
                # save DEC model checkpoints
                self.DEC.save(self.path+'/DEC_model_'+'.h5')
                return self.y_pred

            # update (or initialize) probability distributions and propagate weight changes
            # from DEC model to encoder.
            if iteration % update_interval == 0:
                self.q = self.DEC.predict(X, verbose=0)
                self.p = self.p_mat(self.q)

                y_pred = self.q.argmax(1)
                print(y_pred)
                print("yo")
                print(self.y_pred)
                delta_label = ((y_pred == self.y_pred).sum().astype(np.float32) / y_pred.shape[0])
                y=None
                if y is not None:
                    print(y)
                    acc = self.cluster_acc(y, y_pred)[0]
                    self.accuracy.append(acc)
                    print('Iteration '+str(iteration)+', Accuracy '+str(np.round(acc, 5)))
                else:
                    print(str(np.round(delta_label*100, 5))+'% change in label assignment')

                if delta_label < tol:
                    print('Reached tolerance threshold. Stopping training.')
                    train = False
                    continue
                else:
                    self.y_pred = y_pred
                
                if delta_label >0.996:
                    contator=contator+1
                
                for i in range(len(self.encoder.layers)):
                    self.encoder.layers[i].set_weights(self.DEC.layers[0].layers[i].get_weights())
                self.cluster_centres = self.DEC.layers[-1].get_weights()[0]

            # train on batch
            sys.stdout.write('Iteration %d, ' % iteration)
            if (index+1)*self.batch_size > X.shape[0]:
                loss = self.DEC.train_on_batch(X[index*self.batch_size::], self.p[index*self.batch_size::])
                index = 0
                sys.stdout.write('Loss %f' % loss)
            else:
                loss = self.DEC.train_on_batch(X[index*self.batch_size:(index+1) * self.batch_size],
                                               self.p[index*self.batch_size:(index+1) * self.batch_size])
                sys.stdout.write('Loss %f' % loss)
                index += 1

            # save intermediate
            iteration += 1
            sys.stdout.flush()
        return
