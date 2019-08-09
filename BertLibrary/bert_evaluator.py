from .processors.Label2TextProcessor import Label2TextProcessor
import os

import tensorflow as tf
from BertLibrary.bert.run_classifier import *

class BertEvaluator:

    def __init__(self, model, iter_steps=1000):
        self.model = model
        self.processor = Label2TextProcessor(self.model.max_seq_len)
        self.logging_hook = LoggingSessionHook(self.model, iter_steps)

    def convert_features(self, data_path, output_file):
        examples = self.processor.file_get_examples(data_path)
        file_based_convert_examples_to_features(examples,
                                                self.model.labels,
                                                self.model.max_seq_len,
                                                self.model.tokenizer,
                                                output_file)

    def evaluate(self, X, y, checkpoint=None):
        test_examples = self.processor.get_examples(X, y)

        test_features = convert_examples_to_features(test_examples,
            self.model.labels,
            self.model.max_seq_len,
            self.model.tokenizer)

        test_input_fn = input_fn_builder(
            features=test_features,
            seq_length=self.model.max_seq_len,
            is_training=False,
            drop_remainder=False)

        self.model.estimator.evaluate(
          test_input_fn, checkpoint_path=checkpoint, hooks=[self.logging_hook])

    def evaluate_from_file(self, data_path, checkpoint=None):
        test_file = os.path.join(data_path, 'test.tsv')
        processed_test_file = os.path.join(data_path, 'test.tf-record')

        if not os.path.exists(test_file) and not os.path.exists(processed_test_file):
            raise 'test file missing'

        if not os.path.exists(processed_test_file):
            self.convert_features(test_file, processed_test_file)

        eval_input_fn = file_based_input_fn_builder(
            input_file=processed_test_file,
            seq_length=self.model.max_seq_len,
            is_training=False,
            drop_remainder=False)

        self.model.estimator.evaluate(
          eval_input_fn, checkpoint_path=checkpoint, hooks=[self.logging_hook])



class LoggingSessionHook(tf.train.SessionRunHook):

    def __init__(self, model, iter_steps):
        self.model = model
        self.iter_steps = iter_steps

    # init ops
    def begin(self):
        self.iterations = 0

    # print every k iteration evaluations steps
    def after_run(self, run_context, run_values):
        self.iterations += 1

        if self.iterations % self.iter_steps == 0:
            tf.logging.info(
              'Reached iteration %s, processed %s sentences', 
              self.iterations, self.iterations * self.model.batch_size)

