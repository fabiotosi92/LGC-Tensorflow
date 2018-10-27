import tensorflow as tf

class Dataloader(object):

    def __init__(self, file, image_height, image_width, is_training):
        self.file = file
        self.image_height = image_height
        self.image_width = image_width
        self.is_training = is_training

        self.left = None
        self.disp  = None
        self.gt = None

        input_queue = tf.train.string_input_producer([self.file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)
        split_line = tf.string_split([line], ';').values

        if is_training == 'True':
            self.left = tf.cast(self.read_image(split_line[0], [None, None, 3]), tf.float32)
            self.disp = tf.cast(self.read_image(split_line[1], [None, None, 1], dtype=tf.uint16), tf.float32)/256.0
            self.gt = tf.cast(self.read_image(split_line[2], [None, None, 1], dtype=tf.uint16), tf.float32)/ 256.0
            self.left_filename = split_line[0]
            self.disp_filename = split_line[1]
            self.gt_filename = split_line[2]
        else:
            self.left = tf.stack([tf.cast(self.read_image(split_line[0], [None, None, 3]), tf.float32)], 0)
            self.disp = tf.stack([tf.cast(self.read_image(split_line[1], [None, None, 1], dtype=tf.uint16), tf.float32)], 0)/256.0
            self.disp_filename = split_line[0]
            self.left_filename = split_line[1]

    def get_patches(self, patch_size, threshold, loc = None, glob = None):
        left_list = []
        disp_list = []
        gt_list = []

        left_list.append(self.left)
        disp_list.append(self.disp)
        gt_list.append(self.gt)

        ksizes = [1, patch_size, patch_size, 1]
        strides = [1, 1, 1, 1]
        rates = [1, 1, 1, 1]
        padding = 'VALID'

        left_patches = tf.reshape(
            tf.extract_image_patches(left_list, ksizes, strides, rates, padding), [-1, patch_size, patch_size, 3])

        disp_patches = tf.reshape(
            tf.extract_image_patches(disp_list, ksizes, strides, rates, padding), [-1, patch_size, patch_size, 1])

        gt_patches = tf.reshape(
            tf.extract_image_patches(gt_list, ksizes, strides, rates, padding), [-1, patch_size, patch_size, 1])

        mask = gt_patches[:, int(patch_size / 2):int(patch_size / 2) + 1, int(patch_size / 2):int(patch_size / 2) + 1,:] > 0
        valid = tf.tile(mask, [1, patch_size, patch_size, 1])

        left_patches = tf.reshape(tf.boolean_mask(left_patches, tf.concat([valid, valid, valid], axis=3)),
                                  [-1, patch_size, patch_size, 3])
        disp_patches = tf.reshape(tf.boolean_mask(disp_patches, valid), [-1, patch_size, patch_size, 1])
        gt_patches = tf.reshape(tf.boolean_mask(gt_patches, valid), [-1, patch_size, patch_size, 1])

        labels = tf.cast(tf.abs(disp_patches - gt_patches) <= threshold, tf.float32)

        if loc is not None and glob is not None:
            global_list = []
            local_list = []

            global_list.append(loc)
            local_list.append(glob)

            global_patches = tf.reshape(
            tf.extract_image_patches(global_list, ksizes, strides, rates, padding), [-1, patch_size, patch_size, 1])

            local_patches = tf.reshape(
            tf.extract_image_patches(local_list, ksizes, strides, rates, padding), [-1, patch_size, patch_size, 1])

            global_patches = tf.reshape(tf.boolean_mask(global_patches, valid), [-1, patch_size, patch_size, 1])
            local_patches = tf.reshape(tf.boolean_mask(local_patches, valid), [-1, patch_size, patch_size, 1])

            return left_patches, disp_patches, global_patches, local_patches, labels

        return left_patches, disp_patches, labels

    def get_crops(self, crop_height, crop_width, batch_size):
        crops = tf.random_crop(tf.concat([self.left, self.disp, self.gt], -1), [crop_height, crop_width, 5])
        left_image, disp_image, gt_image = tf.split(crops, [3, 1, 1], axis=2)

        min_after_dequeue = 8
        num_threads = 8
        capacity = min_after_dequeue + 4 * batch_size
        left_image_batch, disp_image_batch, gt_image_batch = tf.train.shuffle_batch([left_image, disp_image, gt_image],
                                                                               batch_size, capacity,
                                                                               min_after_dequeue, num_threads)
        return left_image_batch, disp_image_batch, gt_image_batch

    def read_image(self, image_path, shape=None, dtype=tf.uint8):
        image_raw = tf.read_file(image_path)
        if dtype == tf.uint8:
            image = tf.image.decode_image(image_raw)
        else:
            image = tf.image.decode_png(image_raw, dtype=dtype)
        if shape is None:
            image.set_shape([None, None, 3])
        else:
            image.set_shape(shape)
        if self.is_training == 'True':
            return tf.image.resize_image_with_crop_or_pad(image, self.image_height, self.image_width)
        return image






