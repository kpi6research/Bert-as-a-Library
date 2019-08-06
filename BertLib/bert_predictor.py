import tensorflow as tf
import math
import numpy as np


class BertPredictor:

    def __init__(self, model_estimator, processor, config):
        self.processor = processor
        self.predictor = tf.contrib.predictor.from_estimator(
            model_estimator, processor.serving_input_receiver_fn(), config=config)
        self.bs = self.processor.batch_size

    def __call__(self, sentences):
        return self.predict_key(sentences, self.processor.key)

    def predict_key(self, sentences, key):
        iterations = math.ceil(len(sentences) / self.bs)

        predictions = []
        i = 0

        while i < iterations:
            next_batch = sentences[i*self.bs:(i+1)*self.bs]
            next_batch = self.processor.preprocess_sentences(next_batch)
            y_ = self.predictor(next_batch)[key]
            predictions.append(y_)
            i += 1

        return np.vstack(predictions)

    def predict_all_keys(self, sentences):
        iterations = math.ceil(len(sentences) / self.bs)

        predictions = []
        i = 0

        while i < iterations:
            next_batch = sentences[i*self.bs:(i+1)*self.bs]
            next_batch = self.processor.preprocess_sentences(next_batch)
            y_ = self.predictor(next_batch)
            y_ = [v for _, v in y_.items()]
            predictions.append(y_)
            i += 1

        return np.vstack(predictions)
