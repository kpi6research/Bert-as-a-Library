from run_classifier import *
import tokenization

import os


class Label2TextProcessor(DataProcessor):
    """Processor for Bert data set """

    def __init__(self, labels):
      super().__init__()
      self.labels = labels

    def file_get_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(data_dir))

    def get_examples(self, X, y):
        return self._create_examples(zip(y, X))

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if i == 0:
                continue
            guid = str(i)
            text_a = tokenization.convert_to_unicode(line[1])
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
