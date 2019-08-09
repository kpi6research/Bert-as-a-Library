import os
import tensorflow as tf
from BertLibrary.bert_predictor import BertPredictor
from BertLibrary.bert_trainer import BertTrainer
from BertLibrary.bert_evaluator import BertEvaluator

from tensorflow.estimator import Estimator
from tensorflow.estimator import RunConfig

from BertLibrary.bert.run_classifier import *
import BertLibrary.bert.modeling as modeling
import BertLibrary.bert.tokenization as tokenization


class BertModel:

    def __init__(self,
                 model_dir,
                 ckpt_name,
                 do_lower_case,
                 max_seq_len,
                 batch_size,
                 labels,
                 trainable=True,
                 keep_checkpoint_max=5,
                 config=None):
        self.model_dir = model_dir
        self.bert_config, self.vocab_file, \
            self.init_checkpoint = self.get_model_configs(model_dir, ckpt_name)

        self.do_lower_case = do_lower_case
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.processer = None
        self.keep_checkpoint_max = keep_checkpoint_max
        self.labels = labels
        self.config = config if config else None
        self.predictor = None
        self.trainable = trainable

    def build(self, model_fn_args, config_args):
        config = self.get_config(**config_args)
        model_fn = self.get_model_fn(**model_fn_args)

        self.estimator = Estimator(
            model_fn=model_fn,
            config=config,
            params={'batch_size': self.batch_size})

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

    def get_model_configs(self, base_dir, ckpt_name):
        bert_config_file = os.path.join(base_dir, 'bert_config.json')
        vocab_file = os.path.join(base_dir, 'vocab.txt')
        init_checkpoint = os.path.join(base_dir, ckpt_name)
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        return bert_config, vocab_file, init_checkpoint

    def get_config(self, ckpt_output_dir='./output', save_check_steps=1000):

        if not self.config:
            self.config = tf.ConfigProto(device_count={'GPU': 1})
            self.config.gpu_options.allow_growth = True
            self.config.gpu_options.per_process_gpu_memory_fraction = 0.5

        run_config = RunConfig(
            model_dir=ckpt_output_dir,
            session_config=self.config,
            keep_checkpoint_max=self.keep_checkpoint_max,
            save_checkpoints_steps=save_check_steps)

        return run_config

    def get_predictor(self):
        return BertPredictor(self.estimator, self.processer, self.config)

    def get_trainer(self):
        assert self.trainable, 'This model cannot be trained'
        return BertTrainer(self)

    def get_evaluator(self, iter_steps=1000):
        return BertEvaluator(self, iter_steps=iter_steps)

    def get_model_fn(self, *args):
        return NotImplementedError()