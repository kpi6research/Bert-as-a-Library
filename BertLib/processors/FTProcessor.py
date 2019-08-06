import tensorflow as tf
from .Processor import Processor


class FTProcessor(Processor):

    def __init__(self, max_seq_len, tokenizer, batch_size, pred_key):
        super().__init__(max_seq_len, tokenizer, batch_size, pred_key)

    def serving_input_receiver_fn(self):
        features = {
            "label_ids": tf.placeholder(dtype=tf.int64, shape=[None, 1], name='label_ids'),
            "input_ids": tf.placeholder(dtype=tf.int64, shape=[None, self.max_seq_len], name='input_ids'),
            "input_mask": tf.placeholder(dtype=tf.int64, shape=[None, self.max_seq_len], name='input_mask'),
            "segment_ids": tf.placeholder(dtype=tf.int64, shape=[None, self.max_seq_len], name='segment_ids'),
        }

        return tf.estimator.export.build_raw_serving_input_receiver_fn(features)
    
    def preprocess_sentences(self, sentences):
        features = {'input_ids': [],
                    'input_mask': [],
                    'segment_ids': [],
                    'label_ids': []}

        # Get predictions dictionary
        for i, s in enumerate(sentences):
            input_ids, input_mask, segment_ids, label_ids = self.convert_example(
                i, s, self.max_seq_len, self.tokenizer)

            features['input_ids'].append(input_ids)
            features['input_mask'].append(input_mask)
            features['segment_ids'].append(segment_ids)
            features['label_ids'].append(label_ids)

        return features

    def convert_example(self, ex_index, example, max_seq_length,
                        tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        tokens_a = tokenizer.tokenize(example)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return (input_ids, input_mask, segment_ids, [0])
