import argparse
import tensorflow as tf
from model import LGC

parser = argparse.ArgumentParser(description='Argument parser')

"""Arguments related to run mode"""
parser.add_argument('--is_training', dest='is_training', type=str, default='False', help='train, test')

"""Arguments related to training"""
parser.add_argument('--epoch',  dest='epoch', type=int, default=14, help='# of epoch')
parser.add_argument('--image_height', dest='image_height', type=int, default=384, help='# image height')
parser.add_argument('--image_width', dest='image_width', type=int, default=1280, help='# image width')
parser.add_argument('--crop_height', dest='crop_height', type=int, default=256, help='# crop height')
parser.add_argument('--crop_width', dest='crop_width', type=int, default=512, help='# crop width')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=9, help='# images in patches')
parser.add_argument('--dataset',  dest='dataset', type=str, default='../utils/kitti_training_set.txt', help='dataset')
parser.add_argument('--initial_learning_rate', dest='initial_learning_rate', type=float, default=0.003, help='initial learning rate for gradient descent')
parser.add_argument('--threshold', dest='threshold', type=float, default=3, help='disparity error if absolute difference between disparity and groundtruth > threshold')
parser.add_argument('--late_fusion', help='LFM as local network', action='store_true')

"""Arguments related to models"""
parser.add_argument('--model', dest='model', type=str, default='', help='CCNN, EFN, LFM, ConfNet, LGC')

"""Arguments related to monitoring and outputs"""
parser.add_argument('--log_directory', dest='log_directory', type=str, default='../log', help='directory to save checkpoints and summaries')
parser.add_argument('--checkpoint_path', dest='checkpoint_path', nargs='*', help='path to a specific checkpoint to load')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=2, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--model_name', dest='model_name', type=str, default='CCNN.model', help='model name')
parser.add_argument('--output_path', dest='output_path', type=str, default='../output/CCNN/', help='model name')

args = parser.parse_args()

def main(_):
    with tf.Session() as sess:
        model = LGC(sess,
                     is_training=args.is_training,
                     epoch=args.epoch,
                     threshold=args.threshold,
                     image_height=args.image_height,
                     image_width=args.image_width,
                     crop_height=args.crop_height,
                     crop_width=args.crop_width,
                     batch_size=args.batch_size,
                     patch_size=args.patch_size,
                     initial_learning_rate=args.initial_learning_rate,
                     model=args.model,
                     model_name=args.model_name,
                     late_fusion=args.late_fusion,
                     dataset=args.dataset
                     )

        if args.is_training == 'True':
            model.train(args)
        else:
            model.test(args)


if __name__ == '__main__':
    tf.app.run()
