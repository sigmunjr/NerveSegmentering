import tensorflow as tf

def meshgrid(height, width):
  with tf.variable_scope('meshgrid'):
    # This should be equivalent to:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    x_t = tf.matmul(tf.ones(shape=tf.pack([height, 1])),
                    tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                    tf.ones(shape=tf.pack([1, width])))

    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))

    ones = tf.ones_like(x_t_flat)
    grid = tf.concat(0, [x_t_flat, y_t_flat, ones])
    return grid

def repeat(x, n_repeats):
  with tf.variable_scope('repeat'):
    rep = tf.transpose(
      tf.expand_dims(tf.ones(shape=tf.pack([n_repeats, ])), 1), [1, 0])
    rep = tf.cast(rep, 'int32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])


def interpolate(im, x, y, out_size):
  with tf.variable_scope('interpolate'):
    # constants
    num_batch = tf.shape(im)[0]
    height = tf.shape(im)[1]
    width = tf.shape(im)[2]
    channels = tf.shape(im)[3]

    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    out_height = out_size[0]
    out_width = out_size[1]
    zero = tf.zeros([], dtype='int32')
    max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
    max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

    # scale indices from [-1, 1] to [0, width/height]
    x = (x + 1.0) * (width_f) / 2.0
    y = (y + 1.0) * (height_f) / 2.0

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    dim2 = width
    dim1 = width * height
    base = repeat(tf.range(num_batch) * dim1, out_height * out_width)
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = tf.reshape(im, tf.pack([-1, channels]))
    im_flat = tf.cast(im_flat, 'float32')
    Ia = tf.gather(im_flat, idx_a)
    Ib = tf.gather(im_flat, idx_b)
    Ic = tf.gather(im_flat, idx_c)
    Id = tf.gather(im_flat, idx_d)

    # and finally calculate interpolated values
    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')
    wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
    wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
    wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
    wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
    output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    return output