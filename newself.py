# example of semi-supervised gan for mnist
import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Activation
from matplotlib import pyplot
from keras import backend
from keras.models import Sequential,Model
from operator import add
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical 
import random
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input, Reshape, UpSampling2D, InputLayer, Lambda, ZeroPadding2D, Cropping2D, Conv2DTranspose, BatchNormalization
#from keras.utils import np_utils, to_categorical
from keras.losses import binary_crossentropy

from keras.losses import mse, binary_crossentropy
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
from scipy.stats import entropy

 


tf.keras.backend.set_floatx('float64')

def custom_activation(output):
	logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
	result = logexpsum / (logexpsum + 1.0)
	return result
 
def custome_activation2(output):
	result=backend.exp(output/20) / tf.reduce_sum(backend.exp(output/20)+1)
	return result

def custome_activation3(output):
	result=backend.exp(output) / tf.reduce_sum(backend.exp(output)+1)
	return result

def define_d_model1(in_shape,n_classes):
	# image input
	in_image = Input(shape=in_shape)
	# downsample
	fe = Conv2D(256, (3,3), strides=(2,2), padding='same')(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(256, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Conv2D(256, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	fe=Dense(100)(fe)
	# output layer nodes
	fe = Dense(n_classes)(fe)
	# supervised output
	c_out_layer = Lambda(custome_activation3)(fe)
	# define and compile supervised discriminator model
	c_model2 = Model(in_image, c_out_layer)
	cc_out_layer = Lambda(custome_activation2)(fe)
	cc_model2=Model(in_image, cc_out_layer)
	d_out_layer = Lambda(custom_activation)(fe)
	# define and compile unsupervised discriminator model
	d_model2 = Model(in_image, d_out_layer)
	return c_model2,d_model2,cc_model2

# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(28,28,1), n_classes=10):

	c_model,d_model,cc_model=define_d_model1(in_shape,n_classes)
	c_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
	cc_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
	d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

	return d_model, c_model,cc_model
 
def vae_loss(input_img, output):
    
   ## total_loss = binary_crossentropy(K.flatten(input_img), K.flatten(output)) 
	feature_mean_real = tf.reduce_mean(output, axis=0)
	feature_mean_fake = tf.reduce_mean(input_img, axis=0)
	# L1 distance of features is the loss for the generator
	loss_g = tf.reduce_mean(tf.abs(feature_mean_real - feature_mean_fake))
	return loss_g


# define the standalone generator model
def define_generator(latent_dim):
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, 128))(gen)
	# upsample to 14x14
	gen = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	# define model
	model = Model(in_lat, out_layer)
	model.compile(loss=vae_loss,optimizer=Adam(lr=0.0002, beta_1=0.5))

	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    d_model.trainable = True
    return model
 
# load the images
# load the images
def load_real_samples():
	# load dataset
	(trainX, trainy), (_, _) = mnist.load_data()
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return [X, trainy]


# load the images
def load_test_samples():
	# load dataset
	(_, _),(trainX, trainy) = mnist.load_data()
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return [X, trainy]



# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict(z_input)
	# create class labels
	y = zeros((n_samples, 1))
	return images, y,z_input

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	z_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = z_input.reshape(n_samples, latent_dim)
	return z_input

def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

# size of the latent space
latent_dim = 100
# create the generator


#clients
clientsnum=4
g_model=[None] * clientsnum
d_model=[None] * clientsnum
c_model=[None] * clientsnum
cc_model=[None] * clientsnum
gan_model=[None] * clientsnum

for i in range(clientsnum):
    # create the discriminator models
    d_model[i], c_model[i],cc_model[i]= define_discriminator()
    #plot_model(c_model[i], to_file='c_model_plot.png', show_shapes=True, show_layer_names=True)
    #plot_model(d_model2[i], to_file='d_model_plot.png', show_shapes=True, show_layer_names=True)
    g_model[i] = define_generator(latent_dim) 
    # create the gan
    gan_model[i] = define_gan(g_model[i], d_model[i])
    plot_model(g_model[i], to_file='model_plot1.png', show_shapes=True, show_layer_names=True)
    plot_model(d_model[i], to_file='model_plot2.png', show_shapes=True, show_layer_names=True)

print(gan_model[0].count_params())

# load image data
dataset = load_real_samples()


#testX,testy=load_test_samples()
#print(testX.shape)
X,y=dataset

aprin = [None]*clientsnum
bprin = [None]*clientsnum
aprin_sup = [None]*clientsnum
bprin_sup = [None]*clientsnum

for i in range(clientsnum):
    aprin[i]=[]
    bprin[i]=[]
    aprin_sup[i]=[]
    bprin_sup[i]=[]

len1=len(X)
frac=(int)(len1/(clientsnum))

for i in range(len1):
    flag=False
    while flag==False:
        candidatemachin=y[i]%clientsnum
        rand=random.random()
        if rand<0.2:
            targetmachine=candidatemachin
        else:
            targetmachine=random.randint(0,clientsnum-1)
        if len(aprin[targetmachine])<len1:
            aprin[targetmachine].append(X[i])
            bprin[targetmachine].append(y[i])
            flag=True
for i in range(clientsnum):
    aprin[i]=np.array(aprin[i])
    bprin[i]=np.array(bprin[i])
    aprin_sup[i]=aprin[i][0:int(200/clientsnum)]
    bprin_sup[i]=bprin[i][0:int(200/clientsnum)]

epochs=100
batchsize=32
#batchnum= (int)(frac/batchsize)
batchnum=100

half_batch=batchsize/2

c_loss=[1]*clientsnum
g_loss=[1]*clientsnum
testX,testy=load_test_samples()
categorical_labels_test = to_categorical(testy, num_classes=10)
file_name = 'accuracies.txt'
f = open(file_name, 'a+')  # open file in write mode
for e in range(epochs):
    g_model[i].save("gen%d"%i)
    d_model[i].save("desc%d"%i)
    _, accuracy = c_model[0].evaluate(testX, categorical_labels_test)
    print('Accuracy: %.2f ' % (accuracy*100))
    f.write('Accuracy: %.2f ' % (accuracy*100))
    for b in range(batchnum):
        #*******
        #test_latent=generate_latent_points(latent_dim,batchsize );
        #test_img_gen=[None]*clientsnum
        #for i in range(0,clientsnum):
        #  test_img_gen[i]=g_model[i].predict(test_latent)
        #test_img_gen_avg=np.average(test_img_gen,axis=0)
        #*******
        g_loss=[1/(x+0.001) for x in g_loss]
        norm=np.linalg.norm(g_loss)
        g_loss = (g_loss)/norm
        fake_images,_,latent=generate_fake_samples(g_model[0],latent_dim,(int)(batchsize/clientsnum))
        y_fake=[None]*clientsnum
        z_fake=[None]*clientsnum
        ent=[None]*clientsnum
        for i in range(1,clientsnum):

            a,_,bb=generate_fake_samples(g_model[i],latent_dim,(int)(batchsize/clientsnum))
            fake_images=np.concatenate((fake_images, a),axis=0)
            latent=np.concatenate((latent, bb),axis=0)
            
        for i in range(clientsnum):
            y_fake[i]=cc_model[i].predict(fake_images)
            ent[i]=entropy(y_fake[i],axis=-1)
        c_loss=[1/(x+0.001) for x in ent]
        norm=np.linalg.norm(c_loss)
        c_loss = (c_loss)/norm
        aw = np.array(c_loss)
        bw=np.repeat(aw[ :, :,np.newaxis], 10, axis=-1)
        y_avg=np.average(y_fake,axis=0,weights=bw)

        array_sum = np.sum(y_avg)
        array_has_nan = np.isnan(array_sum)
        if array_has_nan:
            print("hoohoo")
            print(y_fake)
            print("haha")
            print(bw)
            print("hehe")
            print(fake_images)
            exit() 
        for i in range(clientsnum):
            print("epoch ",e," batch ",b," client ",i)
            [Xsup_real, ysup_real], _ = generate_real_samples([aprin_sup[i],bprin_sup[i]], int(half_batch))
            categorical_labels = to_categorical(ysup_real, num_classes=10)
            c_loss[i], c_acc = c_model[i].train_on_batch(Xsup_real, categorical_labels)
            cc_model[i].train_on_batch(fake_images, y_avg+0.000001)
			
            # update unsupervised discriminator (d)
            X,y=dataset
            [X_real, _], y_real = generate_real_samples([aprin[i],bprin[i]], int(half_batch))
            d_loss1 = d_model[i].train_on_batch(X_real, y_real)
            #c_model[i].train_on_batch(add_noise(X_real),c_model[i].predict(X_real))
            X_fake, y_fake,_ = generate_fake_samples(g_model[i], latent_dim, int(half_batch))
            d_loss2 = d_model[i].train_on_batch(X_fake, y_fake)
            # update generator (g)
            X_gan, y_gan = generate_latent_points(latent_dim,batchsize ), ones((batchsize, 1))
            g_loss[i] = gan_model[i].train_on_batch(X_gan, y_gan)
            #g_model[i].train_on_batch(test_latent,test_img_gen_avg)
            #if e>20:
            #	g_model[i].train_on_batch(latent,fake_images)
            #gan_model[i].train_on_batch(fake_images,ones((batchsize, 1)))
            #d_model[i].train_on_batch(fake_images,zeros((fake_images.shape[0],1)))
      
f.close()
testX,testy=load_test_samples()
categorical_labels = to_categorical(testy, num_classes=10)
for i in range(clientsnum):
    _, accuracy = c_model[i].evaluate(testX, categorical_labels)
    print('Accuracy: %.2f' % (accuracy*100))