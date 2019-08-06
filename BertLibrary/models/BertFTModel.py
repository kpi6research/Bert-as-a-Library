import BertLibrary.bert.modeling
import sys
import os
import tensorflow as tf

from tensorflow.estimator import EstimatorSpec


from BertLibrary.bert.run_classifier import *
from .BertModel import BertModel
from BertLibrary.processors.FTProcessor import FTProcessor


class BertFTModel(BertModel):

    def __init__(self,
                 model_dir,
                 ckpt_name,
                 labels,
                 num_train_steps,
                 num_warmup_steps,
                 ckpt_output_dir,
                 save_check_steps,
                 do_lower_case,
                 max_seq_len,
                 batch_size,
                 lr=3e-05,
                 config=None):
        super().__init__(
            model_dir=model_dir,
            ckpt_name=ckpt_name,
            do_lower_case=do_lower_case,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            labels=labels,
            config=config)

        model_fn_args = {'bert_config': self.bert_config,
                         'num_labels': len(labels),
                         'init_checkpoint': self.init_checkpoint,
                         'learning_rate': lr,
                         'num_train_steps': num_train_steps,
                         'num_warmup_steps': num_warmup_steps}

        config_args = {'ckpt_output_dir': ckpt_output_dir,
                       'save_check_steps': save_check_steps}

        self.build(model_fn_args, config_args)

        self.processer = FTProcessor(
            max_seq_len, self.tokenizer, batch_size, pred_key='probabilities')


    def get_model_fn(self, bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps):
      """Returns `model_fn` closure for TPUEstimator."""

      def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
          """The `model_fn` for TPUEstimator."""

          tf.logging.info("*** Features ***")
          for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

          input_ids = features["input_ids"]
          input_mask = features["input_mask"]
          segment_ids = features["segment_ids"]
          label_ids = features["label_ids"]
          is_real_example = None
          if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
          else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

          is_training = (mode == tf.estimator.ModeKeys.TRAIN)

          (total_loss, per_example_loss, logits, probabilities) = create_model(
              bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
              num_labels, False)

          tvars = tf.trainable_variables()
          initialized_variable_names = {}

          if init_checkpoint:
            (assignment_map, initialized_variable_names
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

          tf.logging.info("**** Trainable Variables ****")
          for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
              init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

          output_spec = None
          if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

            output_spec = EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)

          elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                }

            eval_metrics_ops = metric_fn(per_example_loss, label_ids, logits, is_real_example)

            output_spec = EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics_ops)
          else:
            output_spec = EstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities})
          return output_spec

      return model_fn
