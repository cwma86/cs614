#! /usr/bin/env python3
import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10

import numpy as np
from matplotlib import pyplot as plt

logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_generator_model(latent_dim=100, number_classes=10):
    input_class = layers.Input(shape=(1,))
    model1 = layers.Embedding(number_classes, 50)(input_class)
    model1 = layers.Dense(4*4)(model1)
    model1 = layers.Reshape((4,4,1))(model1)


    input_vector = layers.Input(shape=(latent_dim,))
    model2 = layers.Dense(
        initial_nodes[0]* initial_nodes[1] * initial_nodes[2],
        input_dim=latent_dim
        )(input_vector)
    model2 = layers.LeakyReLU(alpha=0.2)(model2)
    model2 = layers.Reshape(initial_nodes)(model2)
    merge =  layers.Concatenate()([model2, model1])
    model2 = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')(merge)
    model2 = layers.BatchNormalization()(model2)
    model2 = layers.LeakyReLU(alpha=0.2)(model2)
    model2 = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')(model2)
    model2 = layers.BatchNormalization()(model2)
    model2 = layers.LeakyReLU(alpha=0.2)(model2)
    model2 = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')(model2)
    model2 = layers.BatchNormalization()(model2)
    model2 = layers.LeakyReLU(alpha=0.2)(model2)
    model2 = layers.Conv2D(32, (3,3), padding='same')(model2)
    model2 = layers.BatchNormalization()(model2)
    model2 = layers.LeakyReLU(alpha=0.2)(model2)
    out_layer = layers.Conv2D(3, (3,3), activation='tanh', padding='same')(model2)
    model = tf.keras.Model([input_vector,input_class], out_layer)  
    return model

# Discriminator model
def make_discriminator_model(input_img_shape=(32,32,3), number_classes=10):
    input_class = layers.Input(shape=(1,))
    model1 = layers.Embedding(number_classes, 50)(input_class)
    model1 = layers.Dense(input_img_shape[0] * input_img_shape[1])(model1)
    model1 = layers.Reshape((input_img_shape[0],input_img_shape[1],1))(model1)

    input_image = layers.Input(shape=input_img_shape)
    merge = layers.Concatenate()([input_image, model1])
    model2 = layers.Conv2D(64, (3,3), padding='same', input_shape=(32,32,3))(merge)
    model2 = layers.BatchNormalization()(model2)
    model2 = layers.LeakyReLU(alpha=0.2)(model2)
    model2 = layers.Conv2D(64, (3,3), strides=(2,2), padding='same')(model2)
    model2 = layers.BatchNormalization()(model2)
    model2 = layers.LeakyReLU(alpha=0.2)(model2)
    model2 = layers.Conv2D(64, (3,3), strides=(2,2), padding='same')(model2)
    model2 = layers.BatchNormalization()(model2)
    model2 = layers.LeakyReLU(alpha=0.2)(model2)
    model2 = layers.Conv2D(64, (3,3), strides=(2,2), padding='same')(model2)
    model2 = layers.BatchNormalization()(model2)
    model2 = layers.LeakyReLU(alpha=0.2)(model2)
    model2 = layers.Conv2D(64, (3,3), strides=(2,2), padding='same')(model2)
    model2 = layers.LeakyReLU(alpha=0.2)(model2)
    model2 = layers.Flatten()(model2)
    model2 = layers.Dropout(0.25)(model2)
    out_layer = layers.Dense(1, activation='sigmoid')(model2)
    model = tf.keras.Model([input_image,input_class], out_layer)  
    return model


def discriminator_loss(output, true_output, loss_func):
    loss = loss_func(true_output, output)
    return loss

def generator_loss(fake_output, loss_func):
    return loss_func(tf.ones_like(fake_output), fake_output)


# Define the training loop
@tf.function
def train_step(generator, generator_optimizer, discriminator, discriminator_optimizer, batch, loss_func):
    noise = [tf.random.normal([len(batch[0]), 100]),
             np.random.randint(10,size=(len(batch[0]),1))]

    with tf.GradientTape() as disc_tape1:
        generated_images = generator(noise, training=True)
        generated_images = tf.image.resize(generated_images, [32,32])
        real_output = discriminator(batch, training=True)
        disc_loss = discriminator_loss(real_output, tf.ones_like(real_output), loss_func)

    gradients_of_discriminator = disc_tape1.gradient(disc_loss, discriminator.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    with tf.GradientTape() as disc_tape2:
        fake_output = discriminator([generated_images,noise[1]], training=True)
        disc_loss = discriminator_loss(fake_output,  tf.zeros_like(fake_output), loss_func)

    gradients_of_discriminator = disc_tape2.gradient(disc_loss, discriminator.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator([generated_images,noise[1]], training=True)
        gen_loss = generator_loss(fake_output, loss_func)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

def evaluate_results(generator, discriminator, train_images, train_labels, epoch=0):
    print(f'Epoch {epoch}')
    # Store the models current state
    generator.save("my_generator")
    discriminator.save("my_discriminator")

    # Generate some image for checking the discriminator accuracy
    number_of_generated_images = 25
    real_images = [train_images[:number_of_generated_images],
                   train_labels[:number_of_generated_images]
    ]

    figure_size = {"length": 5, "width":5}
    fig = plt.figure(figsize=(figure_size["length"],figure_size["width"]))
    for i in range(figure_size["length"]* figure_size["width"]):
        plt.subplot(figure_size["length"], figure_size["width"], i+1)
        plt.title(f"class: {real_images[1][i]}")
        tmp_img = (real_images[0] * 127.5 + 127.5).astype(int) 
        plt.imshow((tmp_img[i, :, :, :]))
        plt.axis('off')

    plt.savefig('real_image_at_epoch_{:04d}.png'.format(epoch))
    plt.close('all')

    noise = [tf.random.normal([number_of_generated_images, 100]),
             np.random.randint(10,size=(number_of_generated_images,1))]
    generated_images = generator.predict(noise)
    sm_generated_images = tf.image.resize(generated_images, [32,32])
    print(f"np.ones((number_of_generated_images,1)) {np.ones((number_of_generated_images,1)).shape}")
    accuracy = tf.keras.metrics.BinaryAccuracy(
            name='binary_accuracy', dtype=None, threshold=0.5
        )
    
    pred_real = discriminator.predict(real_images)
    accuracy.update_state( np.ones((number_of_generated_images,1)), pred_real)
    print(f"real acc {accuracy.result().numpy()}")
    pred_fake = discriminator.predict([sm_generated_images,noise[1]])
    accuracy.update_state( np.zeros((number_of_generated_images,1)), pred_fake)
    print(f"fake acc {accuracy.result().numpy()}")

    # Un-normalize the images
    generated_images = (generated_images * 127.5 + 127.5).astype(int) 

    figure_size = {"length": 5, "width":5}
    fig = plt.figure(figsize=(figure_size["length"],figure_size["width"]))
    for i in range(figure_size["length"]* figure_size["width"]):
        plt.subplot(figure_size["length"], figure_size["width"], i+1)
        plt.title(f"class: {noise[1][i]}")
        plt.imshow((generated_images[i, :, :, :]))
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close('all')

def input():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help'
    )
    parser.add_argument(
        '--gen',
        type=str,
        default=None,
        help="use existing model"
    )
    parser.add_argument(
        '--disc',
        type=str,
        default=None,
        help="use existing model"
    )

    args = parser.parse_args()

    return args

def main(args):
    generator = make_generator_model()
    if args.gen:
        if os.path.isdir(args.gen):
            logger.info(f"loading {args.gen}")
            generator = tf.keras.models.load_model(args.gen)
            generator.summary()
        else:
            logger.error(f"failed to load gen {args.gen}")
            exit(1)
    print(f"generator \n {generator.summary()}")
    tf.keras.utils.plot_model(generator, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)

    discriminator = make_discriminator_model()
    if args.disc:
        if os.path.isdir(args.disc):
            logger.info(f"loading {args.disc}")
            discriminator = tf.keras.models.load_model(args.disc)
            discriminator.summary()
        else:
            logger.error(f"failed to load dis {args.disc}")
            exit(1)
    print(f"discriminator \n {discriminator.summary()}")
    tf.keras.utils.plot_model(discriminator, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
    # Define the loss functions for the generator and discriminator
    loss_func = tf.keras.losses.BinaryCrossentropy()


    # Load and preprocess the cifar10 dataset
    (train_images, train_labels), (_, _) = cifar10.load_data()
    print(f"train_labels {train_labels.shape}")
    # scale between -1 and 1
    train_images = (train_images.astype('float32') - 127.5) / 127.5
    print(f"min {train_images.min()} max {train_images.max()}")
    train_images = np.reshape(train_images, (train_images.shape[0], 32, 32, 3))
    # Define the optimizers for the generator and discriminator
    generator_optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.5)

    # Define the batch size and number of training steps
    BATCH_SIZE = 128

    # Train the model
    EPOCHS = 1200
    for epoch in range(EPOCHS):
        for i in range(train_images.shape[0] // BATCH_SIZE):
            batch = [train_images[i*BATCH_SIZE:(i+1)*BATCH_SIZE], 
                     train_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                     ]
            train_step(
                generator, 
                generator_optimizer, 
                discriminator, 
                discriminator_optimizer, 
                batch, 
                loss_func
                )

        if epoch % 10 == 0:
            evaluate_results(generator, discriminator, train_images, train_labels, epoch)
    evaluate_results(generator, discriminator, train_images, train_labels, epoch)

if __name__ == '__main__':
    args = input()
    main(args)
