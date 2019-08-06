from .processors.Label2TextProcessor import Label2TextProcessor
import os

import tensorflow as tf
from BertLibrary.bert.run_classifier import *


class BertTrainer():

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

    def train(self, X, y, steps, X_val=None, y_val=None, eval_cooldown=600):
        train_examples = self.processor.get_examples(X, y)

        train_features = convert_examples_to_features(train_examples,
            self.model.labels,
            self.model.max_seq_len,
            self.model.tokenizer)

        train_input_fn = input_fn_builder(
            features=train_features,
            seq_length=self.model.max_seq_len,
            is_training=True,
            drop_remainder=False)

        if X_val and y_val:
            dev_examples = self.processor.get_examples(X_val, y_val)

            dev_features = convert_examples_to_features(dev_examples,
                self.model.labels,
                self.model.max_seq_len,
                self.model.tokenizer)

            dev_input_fn = input_fn_builder(
                features=dev_features,
                seq_length=self.model.max_seq_len,
                is_training=True,
                drop_remainder=False)
              
            self.__train_and_evaluate(train_input_fn, dev_input_fn, steps, eval_cooldown)
        
        else:
            self.model.estimator.train(input_fn=train_input_fn, max_steps=steps)

    def __train_and_evaluate(self, train_input_fn, dev_input_fn, steps, eval_cooldown):
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=dev_input_fn, throttle_secs=eval_cooldown)

        tf.estimator.train_and_evaluate(self.model.estimator, train_spec, eval_spec)


    def train_from_file(self, data_path, steps, eval_cooldown=600):
        train_file = os.path.join(data_path, 'train.tsv')
        dev_file = os.path.join(data_path, 'dev.tsv')
        processed_train_file = os.path.join(data_path, 'train.tf-record')
        processed_dev_file = os.path.join(data_path, 'dev.tf-record')

        if not (os.path.exists(train_file) and os.path.exists(dev_file)) :
            raise 'train and/or dev file missing'

        if not os.path.exists(processed_train_file):
            self.convert_features(train_file, processed_train_file)

        if not os.path.exists(processed_dev_file):
            self.convert_features(dev_file, processed_dev_file)

        train_input_fn = file_based_input_fn_builder(
            input_file=processed_train_file,
            seq_length=self.model.max_seq_len,
            is_training=True,
            drop_remainder=True)

        eval_input_fn = file_based_input_fn_builder(
            input_file=processed_dev_file,
            seq_length=self.model.max_seq_len,
            is_training=True,
            drop_remainder=True)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=eval_cooldown)

        tf.estimator.train_and_evaluate(self.model.estimator, train_spec, eval_spec)
