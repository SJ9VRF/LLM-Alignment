from transformers import pipeline
from tqdm import tqdm
import torch

class PPOTrainer:
    def __init__(self, model, ref_model, tokenizer, dataset, config, sentiment_model):
        self.trainer = BasePPOTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            config=config
        )
        self.dataset = dataset
        self.sentiment_model = sentiment_model
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.ref_model.to(self.device)

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for step, batch in enumerate(tqdm(self.dataset)):
                outputs = self.run_training_step(batch)
                self.log_outputs(outputs, step)
            self.save_model(epoch)

    def run_training_step(self, batch):
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)

        # Generate response using the current policy
        responses = self.model.generate(input_ids, max_length=self.config.max_length)

        # Evaluate responses using the sentiment model
        decoded_responses = [self.tokenizer.decode(r, skip_special_tokens=True) for r in responses]
        sentiments = self.sentiment_model.get_sentiment(decoded_responses)

        # Calculate rewards based on sentiment outputs
        rewards = self.calculate_rewards(sentiments)

        # Perform a PPO update
        loss = self.trainer.step(input_ids, responses, rewards)
        return {'loss': loss, 'rewards': rewards.mean().item()}

    def calculate_rewards(self, sentiments):
        # Assume sentiment outputs are logits of positive class
        return torch.tensor([s['score'] for s in sentiments], device=self.device)

    def log_outputs(self, outputs, step):
        print(f"Step {step}, Loss: {outputs['loss']:.4f}, Average Reward: {outputs['rewards']:.4f}")

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), f'model_epoch_{epoch}.pth')
        print(f"Model saved: model_epoch_{epoch}.pth")

