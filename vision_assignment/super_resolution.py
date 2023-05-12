#! /usr/bin/env python3
import argparse
import logging
import os
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_image(image, filename):
  """
    Saves unscaled Tensor Images.
    Args:
      image: 3D image tensor. [height, width, channels]
      filename: Name of the file to save.
  """
  if not isinstance(image, Image.Image):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  image.save("%s" % filename)
  print("Saved as %s" % filename)

def input():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help'
    )
    parser.add_argument(
        'gen',
        type=str,
        help="generator"
    )

    args = parser.parse_args()

    return args

def preprocess_image(image_path):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = tf.image.decode_image(tf.io.read_file(image_path))
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)


def main(args):
    generator = None
    if os.path.isdir(args.gen):
        logger.info(f"loading {args.gen}")
        generator = tf.keras.models.load_model(args.gen)
    else:
        logger.error(f"failed to load gen {args.gen}")
        exit(1)
    number_of_generated_images = 25
    noise = tf.random.normal([number_of_generated_images, 100])
    generated_images = generator.predict(noise)
    generated_images = (generated_images * 127.5 + 127.5).astype(int)
    SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
    model = hub.load(SAVED_MODEL_PATH)

    for count, image in enumerate(generated_images):
       image_path = f"image_{count}.jpg"
       save_image(image, image_path)
       hr_image = preprocess_image(image_path)
       fake_image = model(hr_image)
       image_path = f"image_{count}.jpg"
       save_image(tf.squeeze(fake_image), image_path)

if __name__ == '__main__':
    args = input()
    main(args)
