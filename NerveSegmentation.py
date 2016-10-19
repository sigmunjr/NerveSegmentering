import os
import resnet
import tensorflow as tf
from ReadNeveData import getFilenamesFromFolder, getProduceresForFilenames, getImagesForProducer


is_training = tf.Variable(True, dtype=tf.bool,
                          name='is_training')

def dilutedConv(x, filters_out, ksize=3, stride=1):
  shape = tf.shape(x)
  shape_list = x.get_shape().as_list()
  filters_in = shape_list[-1]

  weights = tf.get_variable('weights_transpose', [3, 3, filters_in, filters_out],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d(
                              dtype=x.dtype))
  return tf.nn.atrous_conv2d(x, weights, 2, padding='SAME', name='DilutedConv')


def deconv(x, filters_out, ksize=3, stride=1):
  with tf.variable_scope('deconv'):
    shape = tf.shape(x)
    shape_list = x.get_shape().as_list()
    filters_in = shape_list[-1]

    weights = tf.get_variable('weights_transpose', [3, 3, filters_out, filters_in],
                              initializer=tf.contrib.layers.xavier_initializer_conv2d(
                                dtype=x.dtype))  # weight_variable(, scale=7.07)
    out_size = tf.pack([shape[0], shape[1] * stride, shape[2] * stride, filters_out])
    upscaled = tf.nn.conv2d_transpose(x, weights, out_size, [1, stride, stride, 1])
    upscaled.set_shape([shape_list[0], shape_list[1] * stride, shape_list[2] * stride, filters_out])
    return upscaled


def depth_conv(x, filters_out):
  with tf.variable_scope('depth_conv'):
    x_conv = resnet._conv(x, filters_out, ksize=1)
    x_conv = resnet._bn(x_conv, is_training)
    return resnet._relu(x_conv)


def nerveSegmentNet(x):
  with tf.variable_scope('Match'):
    with tf.variable_scope('f_16'):
      x = resnet.stack(x, 3, 16, bottleneck=False, is_training=is_training, stride=2)
    with tf.variable_scope('f_64'):
      x = resnet.stack(x, 3, 64, bottleneck=False, is_training=is_training, stride=1)
    with tf.variable_scope('f_128'):
      x = resnet.stack(x, 6, 128, bottleneck=False, is_training=is_training, stride=1)
      B, N, M, C = x.get_shape().as_list()
      x = tf.image.resize_bilinear(x, [N*2, M*2])
    with tf.variable_scope('f_up_128'):
      x = resnet.stack(x, 2, 128, bottleneck=False, is_training=is_training, stride=1)
      x = depth_conv(x, 2)
    return x
    # x = tf.concat(3, [img, patch])
    with tf.variable_scope('size112'):
      x = resnet.stack(x, 3, 16, bottleneck=True, is_training=is_training, stride=2)
    with tf.variable_scope('size56'):
      x = resnet.stack(x, 3, 32, bottleneck=True, is_training=is_training, stride=2)
    with tf.variable_scope('size28'):
      x = resnet.stack(x, 3, 64, bottleneck=True, is_training=is_training, stride=2)
    with tf.variable_scope('size14'):
      x = resnet.stack(x, 3, 64, bottleneck=True, is_training=is_training, stride=2)
    with tf.variable_scope('up_size28'):
      x = resnet.stack(x, 3, 64, bottleneck=True, is_training=is_training, stride=2, _conv=deconv)
    with tf.variable_scope('up_size56'):
      x = resnet.stack(x, 3, 32, bottleneck=True, is_training=is_training, stride=2, _conv=deconv)
    with tf.variable_scope('up_size112'):
      x = resnet.stack(x, 2, 32, bottleneck=True, is_training=is_training, stride=2, _conv=deconv)
    with tf.variable_scope('up_size224'):
      x = resnet.stack(x, 2, 32, bottleneck=True, is_training=is_training, stride=2, _conv=deconv)
      x = depth_conv(x, 2)
    return x

def segmentationLoss(x, mask):
  B, N, M, C = x.get_shape().as_list()
  softmax = tf.reshape(tf.nn.softmax(tf.reshape(x, [-1, C])), [B, N, M, C])
  tf.image_summary('Softmax_image', softmax[:, :, :, :1])

  mask = tf.cast(mask, tf.float32)
  cross_entropy = tf.reduce_mean(-mask * tf.log(softmax[:, :, :, :1] + 10e-12) + (mask - 1) * tf.log(softmax[:, :, :, 1:]) + 10e-10)
  return cross_entropy


def predLabel(x):
  B, N, M, C = x.get_shape().as_list()
  softmax = tf.reshape(tf.nn.softmax(tf.reshape(x, [-1, C])), [B, N, M, C])
  return tf.argmax(softmax, 3)


def diceCoef(x, y):
  x_b = tf.equal(x, 0)
  y_b = tf.equal(y, 1)
  intersect = tf.reduce_sum(tf.cast(tf.logical_and(x_b, y_b), tf.float32), reduction_indices=[1, 2])
  norm_sum = tf.reduce_sum(tf.cast(x_b, tf.float32), reduction_indices=[1, 2]) + tf.reduce_sum(tf.cast(y_b, tf.float32), reduction_indices=[1, 2])
  B = x.get_shape().as_list()[0]
  intersect = tf.Print(intersect, [intersect, norm_sum], summarize=16)
  return tf.reduce_mean(tf.select(norm_sum > 0, 2*intersect/norm_sum, tf.constant(1, dtype=tf.float32, shape=[B])))


def addLossAvgToSummary(loss_op):
  ema = tf.train.ExponentialMovingAverage(0.9)
  apply_avg_loss = ema.apply([loss_op])
  tf.add_to_collection(resnet.UPDATE_OPS_COLLECTION, apply_avg_loss)
  loss_avg = ema.average(loss_op)
  tf.scalar_summary('loss_avg', loss_avg)


def getTrainOp(loss_op, learning_rate=0.001):
  tf.scalar_summary('loss', loss_op)

  optimizer = tf.train.AdadeltaOptimizer()
  grads = optimizer.compute_gradients(loss_op)

  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  apply_gradients = optimizer.apply_gradients(grads)
  batchnorm_updates = tf.get_collection(resnet.UPDATE_OPS_COLLECTION)
  batchnorm_updates_op = tf.group(*batchnorm_updates)
  return tf.group(apply_gradients, batchnorm_updates_op)


def loadModelFromFile(sess, load_dir):
  step = 1
  if tf.gfile.Exists(load_dir):
    restorer = tf.train.Saver(tf.all_variables())
    ckpt = tf.train.get_checkpoint_state(load_dir)
    if ckpt is None: return step
    if os.path.isabs(ckpt.model_checkpoint_path):
      restorer.restore(sess, ckpt.model_checkpoint_path)
    else:
      restorer.restore(sess, os.path.join(load_dir,
                                          ckpt.model_checkpoint_path))

    step_str = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    print('Succesfully loaded model from %s at step=%s.' %
          (ckpt.model_checkpoint_path, step_str))
    step = int(step_str) + 1
  return step


def makeSummary(writer, summary_op, step, sess):
  summary_str = sess.run(summary_op)
  writer.add_summary(summary_str, step)
  writer.flush()

def test():

  fnames = getFilenamesFromFolder('/home/sigmund/Downloads/train/')
  load_dir = train_dir = '/usr/local/models/NerveSeg4'
  load_old = True
  validation_N = 500

  producers = getProduceresForFilenames(fnames[-validation_N:])
  img, mask = getImagesForProducer(producers)
  img_val, mask_val = getImagesForProducer(getProduceresForFilenames(fnames[:-validation_N]), distort=False)

  tf.image_summary('Img', img)
  tf.image_summary('Test_Img', img_val)
  tf.image_summary('Mask', mask)
  tf.image_summary('Mask_val', mask_val)


  x = nerveSegmentNet(img)
  loss_op = segmentationLoss(x, mask)
  addLossAvgToSummary(loss_op)
  train_op = getTrainOp(loss_op)

  pred = predLabel(x)
  tf.image_summary('Pred', tf.cast(tf.expand_dims(pred, 3), tf.float32))
  dice_coef = diceCoef(pred, mask[:, :, :, 0])

  tf.scalar_summary('Dice_coef', dice_coef)

  summary_op = tf.merge_all_summaries()
  sess = tf.Session()
  saver = tf.train.Saver(tf.all_variables(), max_to_keep=10, keep_checkpoint_every_n_hours=2)

  sess.run(tf.initialize_all_variables())
  step = 1 if not load_old else loadModelFromFile(sess, load_dir)
  summary_writer = tf.train.SummaryWriter(train_dir,
                                          graph=sess.graph)
  tf.train.start_queue_runners(sess)

  loss_sum = 0; dice_sum = 0
  cnt = 1
  for step in range(step, 10000000):
    if step % 100 == 0:
      makeSummary(summary_writer, summary_op, step, sess)
      i_val, l_val = sess.run([img_val, mask_val])
      loss_test, dice_test = sess.run([loss_op, dice_coef], feed_dict={img: i_val, mask: l_val, is_training: False})
      dice_sum += dice_test
      loss_sum += loss_test
      print '\t\tTEST', loss_test, dice_test, loss_sum/cnt, dice_sum/cnt
      cnt += 1



def train():
  fnames = getFilenamesFromFolder('/home/sigmund/Downloads/train/')
  load_dir = train_dir = '/usr/local/models/NerveSeg4'
  load_old = True
  validation_N = 500

  producers = getProduceresForFilenames(fnames[-validation_N:])
  img, mask = getImagesForProducer(producers, distort=True)
  img_val, mask_val = getImagesForProducer(getProduceresForFilenames(fnames[:-validation_N]), distort=False)

  tf.image_summary('Img', img)
  tf.image_summary('Test_Img', img_val)
  tf.image_summary('Mask', mask)
  tf.image_summary('Mask_val', mask_val)


  x = nerveSegmentNet(img)
  loss_op = segmentationLoss(x, mask)
  addLossAvgToSummary(loss_op)
  train_op = getTrainOp(loss_op)

  pred = predLabel(x)
  tf.image_summary('Pred', tf.cast(tf.expand_dims(pred, 3), tf.float32))
  dice_coef = diceCoef(pred, mask[:, :, :, 0])

  tf.scalar_summary('Dice_coef', dice_coef)

  summary_op = tf.merge_all_summaries()
  sess = tf.Session()
  saver = tf.train.Saver(tf.all_variables(), max_to_keep=10, keep_checkpoint_every_n_hours=2)

  sess.run(tf.initialize_all_variables())
  step = 1 if not load_old else loadModelFromFile(sess, load_dir)
  summary_writer = tf.train.SummaryWriter(train_dir,
                                          graph=sess.graph)
  tf.train.start_queue_runners(sess)

  for step in range(step, 10000000):
    loss_val, _ = sess.run([loss_op, train_op])
    if step%20==0:
      loss_val, dice, _ = sess.run([loss_op, dice_coef, train_op])
      print step, ' : ', loss_val, ' DICE:', dice
    else:
      loss_val, _ = sess.run([loss_op, train_op])
    if step % 100 == 0:
      makeSummary(summary_writer, summary_op, step, sess)
      i_val, l_val = sess.run([img_val, mask_val])
      loss_test, dice_test = sess.run([loss_op, dice_coef], feed_dict={img: i_val, mask: l_val, is_training: False})
      print '\t\tTEST', loss_test, dice_test
    if step % 1000 == 0:
      saver.save(sess, train_dir + '/model.ckpt', global_step=step)
    print step, ' : ', loss_val


if __name__ == '__main__':
  train()
  # test()

