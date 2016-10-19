import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 8

def getGaussKernel(sigma=0.5):
  x = np.linspace(-2*sigma, 2*sigma, 21)
  return (np.exp(-(x**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))).reshape((-1, 1, 1, 1))

gaussKernel = tf.constant(getGaussKernel(224*0.08), dtype=tf.float32)

def getFilenamesFromFolder(folder_name, find='jpg'):
  filenames = os.listdir(folder_name)
  only_images = filter(lambda x: x.find(find) > -1, filenames)
  remove_masks = filter(lambda x: x.find('mask') == -1, only_images)
  return [folder_name + f for f in remove_masks]


def getProduceresForFilenames(filenames, mask_names=''):
  mask_filenames = [f.replace('.jpg', '_mask.jpg') for f in filenames]
  filename_tensors = tf.convert_to_tensor(filenames, dtype=tf.string)
  mask_filename_tensors = tf.convert_to_tensor(mask_filenames, dtype=tf.string)
  return tf.train.slice_input_producer([filename_tensors, mask_filename_tensors], capacity=BATCH_SIZE*(3+1))


def getImagesForProducer(producer, compare_size=112, SEED=1, img_size=224, distort=True):
  img = tf.image.decode_jpeg(tf.read_file(producer[0]), channels=1)
  seg = tf.cast(tf.greater(tf.image.decode_jpeg(tf.read_file(producer[1]), channels=1), 0), tf.float32)
  img = tf.image.per_image_whitening(img)

  seg_dtype = tf.float32
  if distort:
    img, seg = distortImageAndSegmentation(img, seg, img_size, seg_dtype, SEED)
    img, seg = distortGrid(img, seg)
    img = tf.expand_dims(img, 2)
    seg = tf.expand_dims(seg, 2)
  else:
    img = tf.image.resize_images(img, (img_size, img_size))
    seg = tf.image.resize_images(seg, (img_size, img_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.cast(img, tf.float32)
    seg = tf.cast(seg, seg_dtype)
  return tf.train.batch([img, seg], BATCH_SIZE, num_threads=6)


def distortImageAndSegmentation(img, seg, img_size, seg_dtype, SEED):
  new_size = tf.random_uniform([1], minval=int(0.6 * img_size), maxval=int(img_size * 1.4), dtype=tf.int32)
  new_aspect = tf.random_uniform([1], minval=0.8, maxval=1.2)
  new_width = tf.cast(tf.cast(new_size, tf.float32) * new_aspect, tf.int32)
  diff = tf.cast(tf.tile(tf.expand_dims(tf.maximum(img_size - new_size[0], 0) / 2, 0), [2]), tf.int32)
  zero_tile = tf.zeros([2], dtype=tf.int32)
  img_seg = tf.pack([img, seg])
  img_seg = tf.pad(img_seg, tf.pack([zero_tile, diff, diff, zero_tile]))
  img_seg = tf.image.resize_nearest_neighbor(img_seg, tf.pack(
    [tf.maximum(new_size[0], img_size), tf.maximum(new_width[0], img_size)]))
  img_seg = tf.random_crop(img_seg, [2, img_size, img_size, 1])
  img, seg = tf.unpack(img_seg)
  img = tf.cast(img, tf.float32)
  seg = tf.cast(seg, seg_dtype)
  img = tf.image.random_flip_left_right(img, seed=SEED)
  seg = tf.image.random_flip_left_right(seg, seed=SEED)
  img = tf.image.random_brightness(img, max_delta=20)
  img = tf.image.random_contrast(img,
                                 lower=0.5, upper=1.4)
  return img, seg


def convertTIFFToJPG(filenames):
  for f in filenames:
    try:
      im = Image.open(f)
      im.thumbnail(im.size)
      im.save(f.replace('tif', 'jpg'), 'JPEG', quality=100)
    except Exception, e:
      print e

def getRandomDistortion(size, alpha=8):
  dx = tf.random_uniform([1, size, size, 1], minval=-1, maxval=1)
  return tf.squeeze(tf.nn.conv2d(tf.nn.conv2d(dx, gaussKernel, strides=(1, 1, 1, 1), padding='SAME'),
                      tf.transpose(gaussKernel, [1, 0, 2, 3]), strides=(1, 1, 1, 1), padding='SAME')*alpha*size, [0])



def distortGrid(img, seg=None, alpha=2):
  im_size = img.get_shape().as_list()
  N = im_size[1]
  x = tf.range(0, im_size[1])
  X, Y = tf.meshgrid(x, x)
  X = tf.reshape(X, [N, N, 1])
  Y = tf.reshape(Y, [N, N, 1])
  dx = getRandomDistortion(im_size[1], alpha=alpha)
  dy = getRandomDistortion(im_size[1], alpha=alpha)
  zero = tf.zeros([], dtype='float32')

  x = tf.clip_by_value(tf.cast(X, tf.float32) + dx, zero, N-1.)
  y = tf.clip_by_value(tf.cast(Y, tf.float32) + dy, zero, N-1.)

  x0 = tf.cast(tf.floor(x), 'int32')
  x1 = x0 + 1
  y0 = tf.cast(tf.floor(y), 'int32')
  y1 = y0 + 1
  x00 = tf.concat(2, [y0, x0])
  x10 = tf.concat(2, [y1, x0])
  x01 = tf.concat(2, [y0, x1])
  x11 = tf.concat(2, [y1, x1])

  out_img = getDistortedImage(img, x, x0, x00, x01, x10, x11, y, y0)
  if seg is not None:
    x_nn = tf.to_int32(tf.round(x))
    y_nn = tf.to_int32(tf.round(y))
    out_seg = tf.gather_nd(tf.squeeze(seg), tf.concat(2, [y_nn, x_nn]))
    return out_img, out_seg
  return out_img



def getDistortedImage(img, x, x0, x00, x01, x10, x11, y, y0):
  img = tf.squeeze(img)
  I00 = tf.gather_nd(img, x00)
  I10 = tf.gather_nd(img, x10)
  I01 = tf.gather_nd(img, x01)
  I11 = tf.gather_nd(img, x11)
  a = tf.squeeze(x - tf.to_float(x0))
  b = 1 - a
  c = tf.squeeze(y - tf.to_float(y0))
  d = 1 - c
  out_img = c * (I00 * a + I10 * b) + d * (I01 * a + I11 * b)
  return out_img


def testDistortGrid():
  zero_1 = np.zeros((14, 14), dtype=np.float32)
  zero_1[0, :] = 1
  zero_1[:, 0] = 1
  zero_big = np.tile(zero_1, (16, 16))[np.newaxis, :, :, np.newaxis]
  zero_l = tf.constant(zero_big)
  sess = tf.Session()
  dgrid = distortGrid(tf.squeeze(zero_l))
  print dgrid

if __name__ == '__main__':
  fnames = getFilenamesFromFolder('/home/sigmund/Downloads/train/')
  producers = getProduceresForFilenames(fnames)
  img, mask = getImagesForProducer(producers)

  sess = tf.Session()
  tf.train.start_queue_runners(sess)

  load_dir = train_dir = '/tmp/models/MatchNet3'
