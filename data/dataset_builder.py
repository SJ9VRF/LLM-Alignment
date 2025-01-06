from datasets import load_dataset
from transformers import AutoTokenizer
from .utils import LengthSampler

class DatasetBuilder:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def build_dataset(self, dataset_name="imdb"):
        ds = load_dataset(dataset_name, split="train")
        ds = ds.filter(lambda x: len(x['review']) > 200, batched=False)
        ds = ds.map(self.tokenize, batched=False)
        ds.set_format(type='torch')
        return ds

    def tokenize(self, sample):
        input_size = LengthSampler(self.config.input_min_length, self.config.input_max_length)
        sample['input_ids'] = self.tokenizer.encode(sample['review'])[:input_size()]
        return sample

