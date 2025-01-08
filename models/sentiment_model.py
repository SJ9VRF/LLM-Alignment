from transformers import pipeline

class SentimentModel:
    def __init__(self, model_name):
        self.sentiment_pipe = pipeline("sentiment-analysis", model=model_name)

    def get_sentiment(self, texts):
        return self.sentiment_pipe(texts)
