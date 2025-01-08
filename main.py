from data.dataset_builder import DatasetBuilder
from models.ppo_trainer import PPOTrainer
from models.sentiment_model import SentimentModel
from transformers import AutoModelForCausalLM
from utils.config import Config

def main():
    config = Config()
    dataset_builder = DatasetBuilder()
    dataset = dataset_builder.build_dataset()

    tokenizer = dataset_builder.tokenizer
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    sentiment_model = SentimentModel("lvwerra/distilbert-imdb")

    trainer = PPOTrainer(model, ref_model, tokenizer, dataset, config)
    trainer.train(epochs=10)

if __name__ == "__main__":
    main()
