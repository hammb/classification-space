import classification_config as config
import argparse
import os


class CommandLineArguments:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-f', '--fold', default=0,
                                 help='The fold number for k-fold-crossval', required=True, type=int)
        self.parser.add_argument('-l', '--load', action='store_true')
        self.parser.add_argument('-r', '--resume', action='store_true')
        self.parser.add_argument('-a', '--augmentation', action='store_false')
        self.parser.add_argument('-p', '--path', default=None,
                                 help='Path to model parameters', required=False, type=str)
        self.parser.add_argument('-t', '--task', default="",
                                 help='Task to train', required=True, type=str)
        self.parser.add_argument('-b', '--batch_size', default=1,
                                 help='Batchsize', required=True, type=int)

        self.args = None

    def parse_args(self):
        self.args = self.parser.parse_args()

        config.FOLD = self.args.fold
        config.LOAD_MODEL = self.args.load
        config.RESUME_TRAINING = self.args.resume
        config.TASK = self.args.task
        config.AUGMENTATION = self.args.augmentation
        config.BATCH_SIZE = self.args.batch_size

        config.TRAIN_DIR = os.path.join(os.environ['cs_data_path'], config.TASK, "train", "all_samples")
        config.NUM_INPUT_CHANNELS = 3 if config.TASK == "mprage_3in" else 1

        # if config.LOAD_MODEL:
        #    config.LOAD_MODEL_PATH = self.args.path
