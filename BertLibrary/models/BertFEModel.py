from BertLibrary.bert.run_classifier import *
import BertLibrary.bert.modeling as modeling

from .BertModel import BertModel
from BertLibrary.processors.FEProcessor import FEProcessor

from tensorflow.estimator import EstimatorSpec

import sys
import os
import tensorflow as tf


class BertFEModel(BertModel):

    def __init__(self,
                 model_dir,
                 ckpt_name,
                 layer,
                 do_lower_case,
                 max_seq_len,
                 batch_size,
                 config=None):
        super().__init__(
            model_dir=model_dir,
            ckpt_name=ckpt_name,
            do_lower_case=do_lower_case,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            labels=[0],
            config=config,
            trainable=False)

        model_fn_args = {'bert_config': self.bert_config,
                         'layer': layer,
                         'init_checkpoint': self.init_checkpoint}
        config_args = {}

        self.build(model_fn_args, config_args)

        self.processer = FEProcessor(
            max_seq_len, self.tokenizer, batch_size, pred_key='predictions')

    def get_model_fn(self,
                     bert_config,
                     layer,
                     init_checkpoint):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" %
                                (name, features[name].shape))

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]

            output_sequence = self.create_fe_model(
                bert_config, layer, input_ids, input_mask, segment_ids)

            tvars = tf.trainable_variables()
            initialized_variable_names = {}

            if init_checkpoint:
                (assignment_map, initialized_variable_names
                 ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

                tf.train.init_from_checkpoint(
                    init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

            output_spec = EstimatorSpec(
                mode=mode,
                predictions={"predictions": output_sequence})
            return output_spec

        return model_fn

    def create_fe_model(self,
                        bert_config,
                        layer,
                        input_ids,
                        input_mask,
                        segment_ids):
        """Creates a classification model."""
        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)

        # In the demo, we are doing a simple classification task on the entire
        # segment.
        #
        # If you want to use the token-level output, use model.get_sequence_output()
        # instead.
        # or get_all_encoder_layers to get all the layers
        #output_sequence = model.get_sequence_output()
        output_sequence = model.all_encoder_layers[layer]
        input_mask = tf.cast(input_mask, tf.float32)
        output_sequence = tf.expand_dims(input_mask, -1) * output_sequence

        return output_sequence
