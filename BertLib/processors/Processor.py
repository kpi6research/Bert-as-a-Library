class Processor():

    def __init__(self, max_seq_len, tokenizer, batch_size, key):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.key = key

        
    def serving_input_receiver_fn(self):
        return NotImplementedError()

    def preprocess_sentences(self, sentences):
        return NotImplementedError()