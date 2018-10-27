import tensorflow as tf
import numpy as np
import ops
import time
import os
import cv2
from dataloader import Dataloader


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


class LGC(object):

    def __init__(self, sess, late_fusion, epoch=14, threshold=3, initial_learning_rate=0.003, image_height=384, image_width=1280, crop_height=256, crop_width=512, 
                  batch_size=128, patch_size=9, is_training=False, model='LGC', model_name='model', dataset='./fileutils/training.txt'):
                                  
        self.sess = sess
        self.is_training = is_training
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.image_height = image_height
        self.image_width = image_width
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.radius = int(patch_size/2)
        self.threshold = threshold
        self.initial_learning_rate = initial_learning_rate
        self.model = model
        self.model_name = model_name
        self.epoch=epoch
        self.model_collection = [self.model_name]
        self.dataset = dataset
        self.late_fusion = late_fusion

        self.build_dataloader()
        self.build_model()

        if self.is_training == 'True':
            self.build_losses()
            self.build_summaries()


    def build_dataloader(self):
        self.dataloader = Dataloader(file=self.dataset, image_height=self.image_height, image_width=self.image_width, is_training=self.is_training)

    def build_model(self):
        if self.is_training=='True': #train
            if self.model == 'LFN' or self.model == 'CNN' or self.model == 'EFN':
                self.left = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, 3], name='left')
                self.disp = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, 1], name='disparity')
                self.gt = tf.placeholder(tf.float32, [self.batch_size, 1, 1, 1], name='gt')
            elif self.model == 'ConfNet':
                self.left, self.disp, self.gt = self.dataloader.get_crops(self.crop_height, self.crop_width, self.batch_size)
            else:
                self.left_full = tf.expand_dims(self.dataloader.left, 0)
                self.disp_full = tf.expand_dims(self.dataloader.disp, 0)
                self.left = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, 3], name='left')
                self.disp = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, 1], name='disparity')
                self.glob = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, 1], name='global')
                self.local = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, 1], name='local')
                self.gt = tf.placeholder(tf.float32, [self.batch_size, 1, 1, 1], name='gt')
        else: #test
            self.left = tf.placeholder(tf.float32, name='left')
            self.disp = tf.placeholder(tf.float32, name='disparity')

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        {'CCNN': self.EFN, 
         'EFN':  self.EFN,
         'LFN':  self.LFN, 
         'ConfNet': self.ConfNet,
         'LGC': self.LGC}[self.model]()

    def EFN(self): #CCNN/EFN

        kernel_size = 3
        filters = 64
        fc_filters = 100

        if self.model == "EFN":
            print(" [*] Building EFN model...")
            nchannels = 4
            model_input = tf.concat([self.disp, self.left], axis=3)
        else: #CCNN
            print(" [*] Building CCNN model...")
            nchannels=1
            if self.model == 'LGC':
                disp = self.disp_full if self.is_training == 'True' else self.disp
                model_input = tf.pad(disp, [[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]])
            else:
                model_input = self.disp

        with tf.variable_scope('CCNN'):
            with tf.variable_scope("conv1"):
                conv1 = ops.conv2d(model_input, [kernel_size, kernel_size, nchannels, filters], 1, True, padding='VALID')

            with tf.variable_scope("conv2"):
                conv2 = ops.conv2d(conv1, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.variable_scope("conv3"):
                conv3 = ops.conv2d(conv2, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.variable_scope("conv4"):
                conv4 = ops.conv2d(conv3, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.variable_scope("fully_connected_1"):
                fc1 = ops.conv2d(conv4, [1, 1, filters, fc_filters], 1, True, padding='VALID')

            with tf.variable_scope("fully_connected_2"):
                fc2 = ops.conv2d(fc1, [1, 1, fc_filters, fc_filters], 1, True, padding='VALID')

            with tf.variable_scope("prediction"):
                if self.model == 'LGC':
                    self.local_prediction = tf.nn.sigmoid(ops.conv2d(fc2, [1, 1, fc_filters, 1], 1, False, padding='VALID'))
                else:
                    self.prediction = ops.conv2d(fc2, [1, 1, fc_filters, 1], 1, False, padding='VALID')

    def LFN(self):
        print(" [*] Building LFN model...")

        kernel_size = 3
        filters = 64
        fc_filters = 100

        if self.model == 'LGC':
            disp, left = (self.disp_full, self.left_full) if self.is_training == 'True' else (self.disp, self.left)
            model_input_disp = tf.pad(disp, [[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]])
            model_input_left = tf.pad(left, [[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]])
        else:
            model_input_disp = self.disp
            model_input_left = self.left

        with tf.variable_scope('LFN'):
            with tf.variable_scope('disparity'):
                with tf.variable_scope("conv1"):
                    conv1_disp = ops.conv2d(model_input_disp, [kernel_size, kernel_size, 1, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv2"):
                    conv2_disp = ops.conv2d(conv1_disp, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv3"):
                    conv3_disp = ops.conv2d(conv2_disp, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv4"):
                    conv4_disp = ops.conv2d(conv3_disp, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.variable_scope('RGB'):
                with tf.variable_scope("conv1"):
                    conv1_left = ops.conv2d(model_input_left, [kernel_size, kernel_size, 3, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv2"):
                    conv2_left = ops.conv2d(conv1_left, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv3"):
                    conv3_left = ops.conv2d(conv2_left, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv4"):
                    conv4_left = ops.conv2d(conv3_left, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.variable_scope("fully_connected_1"):
                fc1 = ops.conv2d(tf.concat([conv4_left, conv4_disp], axis=3), [1, 1, 2 * filters, fc_filters], 1, True, padding='VALID')

            with tf.variable_scope("fully_connected_2"):
                fc2 = ops.conv2d(fc1, [1, 1, fc_filters, fc_filters], 1, True, padding='VALID')

            with tf.variable_scope("prediction"):
                if self.model == 'LGC':
                    self.local_prediction = tf.nn.sigmoid(ops.conv2d(fc2, [1, 1, fc_filters, 1], 1, False, padding='VALID'))
                else:
                    self.prediction = ops.conv2d(fc2, [1, 1, fc_filters, 1], 1, False, padding='VALID')

    def LGC(self):
        print(" [*] Building LGC model...")

        kernel_size = 3
        filters = 64
        fc_filters = 100
        scale=255.0

        self.LFN() if self.late_fusion else self.EFN() 
        self.ConfNet()

        model_input_disp = self.disp
        model_input_local, model_input_global = (self.local, self.glob) if self.is_training == 'True' else (self.local_prediction, self.global_prediction)

        with tf.variable_scope('LGC'):
            with tf.variable_scope('disparity'):

                with tf.variable_scope("conv1"):
                    conv1_disp = ops.conv2d(model_input_disp, [kernel_size, kernel_size, 1, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv2"):
                    conv2_disp = ops.conv2d(conv1_disp, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv3"):
                    conv3_disp = ops.conv2d(conv2_disp, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv4"):
                    conv4_disp = ops.conv2d(conv3_disp, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.variable_scope('local'):
                with tf.variable_scope("conv1"):
                    conv1_local = ops.conv2d(model_input_local*scale, [kernel_size, kernel_size, 1, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv2"):
                    conv2_local = ops.conv2d(conv1_local, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv3"):
                    conv3_local = ops.conv2d(conv2_local, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv4"):
                    conv4_local = ops.conv2d(conv3_local, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.variable_scope('global'):
                with tf.variable_scope("conv1"):
                    conv1_global = ops.conv2d(model_input_global*scale, [kernel_size, kernel_size, 1, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv2"):
                    conv2_global = ops.conv2d(conv1_global, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv3"):
                    conv3_global = ops.conv2d(conv2_global, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

                with tf.variable_scope("conv4"):
                    conv4_global = ops.conv2d(conv3_global, [kernel_size, kernel_size, filters, filters], 1, True, padding='VALID')

            with tf.variable_scope("fully_connected_1"):             
                fc1 = ops.conv2d(tf.concat([conv4_global, conv4_local, conv4_disp], axis=3), [1, 1, 3 * filters, fc_filters], 1, True, padding='VALID')

            with tf.variable_scope("fully_connected_2"):
                fc2 = ops.conv2d(fc1, [1, 1, fc_filters, fc_filters], 1, True, padding='VALID')
                
            with tf.variable_scope("prediction"):
                self.prediction = ops.conv2d(fc2, [1, 1, fc_filters, 1], 1, False, padding='VALID')

    def ConfNet(self):
        print(" [*] Building ConfNet model...")

        kernel_size = 3
        filters = 32

        if (self.model == "ConfNet") or (self.model == "LGC" and self.is_training == 'False'):
            left = self.left
            disp = self.disp
        else:
            left = self.left_full
            disp = self.disp_full

        with tf.variable_scope('ConfNet'):
            with tf.variable_scope('RGB'):
                with tf.variable_scope("conv1"):
                    self.conv1_RGB = ops.conv2d(left, [kernel_size, kernel_size, 3, filters], 1, True, padding='SAME')
                                    
            with tf.variable_scope('disparity'):  
                with tf.variable_scope("conv1"):
                    self.conv1_disparity = ops.conv2d(disp, [kernel_size, kernel_size, 1, filters], 1, True, padding='SAME')
            
            model_input = tf.concat([self.conv1_RGB, self.conv1_disparity], axis=3)
            
            self.net1, self.scale1 = ops.encoding_unit('1', model_input, filters * 2)
            self.net2, self.scale2 = ops.encoding_unit('2', self.net1,   filters * 4)
            self.net3, self.scale3 = ops.encoding_unit('3', self.net2,   filters * 8)
            self.net4, self.scale4 = ops.encoding_unit('4', self.net3,   filters * 16)
            
            self.net5 = ops.decoding_unit('4', self.net4, num_outputs=filters * 8, forwards=self.scale4)
            self.net6 = ops.decoding_unit('3', self.net5, num_outputs=filters * 4, forwards=self.scale3)
            self.net7 = ops.decoding_unit('2', self.net6, num_outputs=filters * 2,  forwards=self.scale2)
            self.net8 = ops.decoding_unit('1', self.net7, num_outputs=filters, forwards=model_input)
                        
            if self.model == 'LGC':
                self.global_prediction = tf.nn.sigmoid(ops.conv2d(self.net8, [kernel_size, kernel_size, filters, 1], 1, False, padding='SAME'))
            else:
                self.prediction = ops.conv2d(self.net8, [kernel_size, kernel_size, filters, 1], 1, False, padding='SAME')

    def build_losses(self):
        with tf.variable_scope('loss'):
            if self.model == 'ConfNet':
                self.mask = tf.cast(tf.not_equal(self.gt, 0.0), dtype=tf.float32)
                self.labels = tf.cast(tf.abs(tf.subtract(self.gt, self.disp)) <= self.threshold, dtype=tf.float32)
                self.loss = tf.losses.sigmoid_cross_entropy(self.labels, self.prediction, self.mask)
            else:
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.gt, logits=self.prediction))

    def train(self, args):
        if self.model == 'LGC':
            self.train_LGC(args)
        elif self.model == 'ConfNet':
            self.train_global(args)
        else:
            self.train_local(args)

    def train_local(self, args):
        print("\n [*] Training....")

        if not os.path.exists(args.log_directory):
            os.makedirs(args.log_directory)

        self.vars = tf.all_variables()
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, var_list=self.vars)
        self.saver = tf.train.Saver()
        self.summary_op = tf.summary.merge_all(self.model_collection[0])
        self.writer = tf.summary.FileWriter(args.log_directory + "/summary/", graph=self.sess.graph)

        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print(" [*] Number of trainable parameters: {}".format(total_num_parameters))

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        print(' [*] Loading training set...')
        patches_left, patches_disp, patches_gt = self.dataloader.get_patches(self.patch_size, self.threshold)
        line = self.dataloader.disp_filename
        num_samples = count_text_lines(self.dataset)

        print(' [*] Training data loaded successfully')
        epoch = 0
        iteration = 0
        lr = self.initial_learning_rate

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
  
        print(" [*] Start Training...")
        while epoch < self.epoch:
            for i in range(num_samples):
                batch_left, batch_disp, batch_gt, filename = self.sess.run([patches_left, patches_disp, patches_gt, line])
                print(" [*] Training image: " + filename)

                step_image = 0
                while step_image < len(batch_disp):
                    offset = (step_image * self.batch_size) % (batch_disp.shape[0] - self.batch_size)
                    batch_reference = batch_left[offset:(offset + self.batch_size), :, :, :]
                    batch_data = batch_disp[offset:(offset + self.batch_size), :, :, :]
                    batch_labels = batch_gt[offset:(offset +  self.batch_size), self.radius:self.radius+1, self.radius:self.radius+1, :]

                    _, loss, summary_str = self.sess.run([self.optimizer, self.loss, self.summary_op], feed_dict={self.left:batch_reference, self.disp:batch_data, self.gt:batch_labels, self.learning_rate: lr})

                    print("Epoch: [%2d]" % epoch + ", Image: [%2d]" % i + ", Iter: [%2d]" % iteration + ", Loss: [%2f]" % loss )
                    self.writer.add_summary(summary_str, global_step=iteration)
                    iteration = iteration + 1
                    step_image = step_image + self.batch_size

            epoch = epoch + 1

            if np.mod(epoch, args.save_epoch_freq) == 0:
                self.saver.save(self.sess, args.log_directory + '/' + self.model_name, global_step=iteration)

            if epoch == 10:
                lr = lr/10

        coord.request_stop()
        coord.join(threads)

    def train_global(self, args):
        print("\n [*] Training....")

        if not os.path.exists(args.log_directory):
            os.makedirs(args.log_directory)

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, var_list=tf.all_variables())
        self.saver = tf.train.Saver()
        self.summary_op = tf.summary.merge_all(self.model_collection[0])
        self.writer = tf.summary.FileWriter(args.log_directory + "/summary/", graph=self.sess.graph)

        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print(" [*] Number of trainable parameters: {}".format(total_num_parameters))

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        print(' [*] Loading training set...')
        line = self.dataloader.disp_filename
        num_samples = count_text_lines(self.dataset)

        steps_per_epoch = np.ceil(num_samples / self.batch_size).astype(np.int32)
        num_total_steps = self.epoch * steps_per_epoch
        lr = self.initial_learning_rate

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print(" [*] Start Training...")
        for step in range(0, num_total_steps):

            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.learning_rate: lr})

            print("Step: [%2d]" % step + "/[%2d]" % num_total_steps + ", Loss: [%2f]" % loss)

            if step % 2 == 0:
                summary_str = self.sess.run(self.summary_op, feed_dict={self.learning_rate: lr})
                self.writer.add_summary(summary_str, global_step=step)

            if step % 5000 == 0:
                self.saver.save(self.sess, args.log_directory + '/' + self.model_name, global_step=step)

            if step == steps_per_epoch * 1000:
                lr = lr/10

        self.saver.save(self.sess, args.log_directory + '/' + self.model_name, global_step=num_total_steps)

        coord.request_stop()
        coord.join(threads)

    def train_LGC(self, args):
        print("\n [*] Training....")

        if not os.path.exists(args.log_directory):
            os.makedirs(args.log_directory)

        self.vars = tf.all_variables()
        self.vars_global = [k for k in self.vars if k.name.startswith('ConfNet')]
        self.vars_local =  [k for k in self.vars if (k.name.startswith('CCNN') or k.name.startswith('LFN'))]
        self.vars_lgc = [k for k in self.vars if k.name.startswith('LGC')]

        self.saver_global = tf.train.Saver(self.vars_global)
        self.saver_local = tf.train.Saver(self.vars_local)
        self.saver = tf.train.Saver(self.vars_lgc)

        self.summary_op = tf.summary.merge_all(self.model_collection[0])
        self.writer = tf.summary.FileWriter(args.log_directory + "/summary/", graph=self.sess.graph)

        total_num_parameters = 0
        for variable in self.vars_lgc:
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print(" [*] Number of trainable parameters: {}".format(total_num_parameters))

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, var_list=self.vars_lgc)

        if args.checkpoint_path:
            self.saver_global.restore(self.sess, args.checkpoint_path[0])
            self.saver_local.restore(self.sess, args.checkpoint_path[1])
            print(" [*] Load model: SUCCESS")
        else:
            print(" [*] Load failed...neglected")
            print(" [*] End Testing...")
            raise ValueError('args.checkpoint_path is None')

        print(' [*] Loading training set...')
        patches_left, patches_disp, patches_global, patches_local, patches_gt = self.dataloader.get_patches(self.patch_size, self.threshold, self.global_prediction[0], self.local_prediction[0])
        left = self.dataloader.left
        disp = self.dataloader.disp
        gt = self.dataloader.gt
        line = self.dataloader.disp_filename
        num_samples = count_text_lines(self.dataset)

        print(' [*] Training data loaded successfully')
        epoch = 0
        iteration = 0
        lr = self.initial_learning_rate

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        left = tf.expand_dims(left, 0)
        disp = tf.expand_dims(disp, 0)
        gt = tf.expand_dims(gt, 0)

        print(" [*] Start Training...")
        while epoch < self.epoch:
            for i in range(num_samples):
                batch_left, batch_disp, batch_gt, batch_global, batch_local, filename  = self.sess.run([patches_left, patches_disp, patches_gt, patches_global, patches_local, line])

                print(" [*] Training image: " + filename)

                step_image = 0
                while step_image < len(batch_disp):
                    offset = (step_image * self.batch_size) % (batch_disp.shape[0] - self.batch_size)
                    batch_reference = batch_left[offset:(offset + self.batch_size), :, :, :]
                    batch_data = batch_disp[offset:(offset + self.batch_size), :, :, :]
                    batch_glob = batch_global[offset:(offset + self.batch_size), :, :, :]
                    batch_loc = batch_local[offset:(offset + self.batch_size), :, :, :]
                    batch_labels = batch_gt[offset:(offset +  self.batch_size), self.radius:self.radius+1, self.radius:self.radius+1, :]

                    _, loss, summary_str = self.sess.run([self.optimizer, self.loss, self.summary_op],
                                                         feed_dict={self.left:batch_reference, self.disp:batch_data, self.glob:batch_glob, self.local:batch_loc, self.gt:batch_labels, self.learning_rate: lr})

                    print("Epoch: [%2d]" % epoch + ", Image: [%2d]" % i + ", Iter: [%2d]" % iteration + ", Loss: [%2f]" % loss )
                    self.writer.add_summary(summary_str, global_step=iteration)
                    iteration = iteration + 1
                    step_image = step_image + self.batch_size

            epoch = epoch + 1

            if np.mod(epoch, args.save_epoch_freq) == 0:
                self.saver.save(self.sess, args.log_directory + '/' + self.model_name, global_step=iteration)

            if epoch == 10:
                lr = lr/10

        coord.request_stop()
        coord.join(threads)

    def test(self, args):
        print("[*] Testing....")

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        if self.model == 'LGC':
            self.vars = tf.all_variables()
            self.vars_global = [k for k in self.vars if k.name.startswith('ConfNet')]
            self.vars_local = [k for k in self.vars if (k.name.startswith('CCNN') or k.name.startswith('LFN')) ]
            self.vars_lgc = [k for k in self.vars if k.name.startswith('LGC')]

            self.saver_global = tf.train.Saver(self.vars_global)
            self.saver_local = tf.train.Saver(self.vars_local)
            self.saver_LGC = tf.train.Saver(self.vars_lgc)

            if args.checkpoint_path[0] and args.checkpoint_path[1] and args.checkpoint_path[2]:
                self.saver_global.restore(self.sess, args.checkpoint_path[0])
                self.saver_local.restore(self.sess, args.checkpoint_path[1])
                self.saver_LGC.restore(self.sess, args.checkpoint_path[2])
                
                print(" [*] Load model: SUCCESS")
            else:
                print(" [*] Load failed...neglected")
                print(" [*] End Testing...")
                raise ValueError('args.checkpoint_path[0] or args.checkpoint_path[1] or args.checkpoint_path[2] is None')
        else:
            self.saver = tf.train.Saver()

            if args.checkpoint_path:
                self.saver.restore(self.sess, args.checkpoint_path[0])
                print(" [*] Load model: SUCCESS")
            else:
                print(" [*] Load failed...neglected")
                print(" [*] End Testing...")
                raise ValueError('args.checkpoint_path is None')

        disp_batch = self.dataloader.disp
        left_batch = self.dataloader.left
        line = self.dataloader.disp_filename
        num_samples = count_text_lines(self.dataset)

        if self.model == 'ConfNet':
            prediction = tf.nn.sigmoid(self.prediction)
        else:
            prediction = tf.pad(tf.nn.sigmoid(self.prediction), tf.constant([[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]]), "CONSTANT")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print(" [*] Start Testing...")
        for step in range(num_samples):
            batch_left, batch_disp, filename = self.sess.run([left_batch, disp_batch, line])
            if self.model == 'ConfNet' or self.model == 'LGC':
                val_disp, hpad, wpad = ops.pad(batch_disp)
                val_left, _, _ = ops.pad(batch_left)

            print(" [*] Test image:" + filename)
            start = time.time()
            if self.model == 'ConfNet' or self.model == 'LGC':
                confidence = self.sess.run(prediction, feed_dict={self.left: val_left, self.disp: val_disp})
                confidence = ops.depad(confidence, hpad, wpad)
            else:
                confidence = self.sess.run(prediction, feed_dict={self.left: batch_left, self.disp: batch_disp})
            current = time.time()
            output_file = args.output_path + filename.strip().split('/')[-1]

            cv2.imwrite(output_file, (confidence[0] * 65535.0).astype('uint16'))
            print(" [*] Confidence prediction saved in:" + output_file)
            print(" [*] Running time:" + str(current - start) + "s")

        coord.request_stop()
        coord.join(threads)

    def build_summaries(self):
        tf.summary.scalar('loss', self.loss, collections=self.model_collection)
        tf.summary.scalar('learning_rate', self.learning_rate, collections=self.model_collection)
        if self.model == 'ConfNet':
            tf.summary.image('left', self.left, collections=self.model_collection)
            tf.summary.image('confidence', tf.nn.sigmoid(self.prediction), collections=self.model_collection)
            tf.summary.image('disparity', self.disp, collections=self.model_collection)
            tf.summary.image('labels', self.labels * self.mask, collections=self.model_collection)


