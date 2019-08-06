from .processors.Label2TextProcessor import Label2TextProcessor
import os

import tensorflow as tf
from run_classifier import *

class BertEvaluator:

    def __init__(self, model):
        self.model = model
        self.processor = Label2TextProcessor(self.model.max_seq_len)


    def convert_features(self, data_path, output_file):
        examples = self.processor.file_get_examples(data_path)
        file_based_convert_examples_to_features(examples,
                                                self.model.labels,
                                                self.model.max_seq_len,
                                                self.model.tokenizer,
                                                output_file)

    def evaluate(self, X, y, checkpoint):
        test_examples = self.processor.get_examples(X, y)

        test_features = convert_examples_to_features(test_examples,
            self.model.labels,
            self.model.max_seq_len,
            self.model.tokenizer)

        test_input_fn = input_fn_builder(
            features=test_features,
            seq_length=self.model.max_seq_len,
            is_training=True,
            drop_remainder=False)

        self.model.estimator.evaluate(
          test_input_fn, checkpoint_path=checkpoint)

    def evaluate_from_file(self, data_path, checkpoint=None):
        test_file = os.path.join(data_path, 'test.tsv')
        processed_test_file = os.path.join(data_path, 'test.tf-record')

        if not os.path.exists(test_file) :
            raise 'test file missing'

        if not os.path.exists(processed_test_file):
            self.convert_features(test_file, processed_test_file)

        eval_input_fn = file_based_input_fn_builder(
            input_file=processed_test_file,
            seq_length=self.model.max_seq_len,
            is_training=True,
            drop_remainder=True)

        self.model.estimator.evaluate(
          eval_input_fn, checkpoint_path=checkpoint)
