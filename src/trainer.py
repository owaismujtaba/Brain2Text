import torch
from src.data_loader import get_data_loaders
from src.models import EEG2WhisperEmbedding

class Trainer:
    def __init__(self, config, logger):
        self.model = EEG2WhisperEmbedding(input_dim=config.model.input_dim, output_dim=config.model.output_dim)
        self.config = config
        self.logger = logger
        if torch.cuda.is_available() and config.training.device == "cuda":
            self.device = torch.device("cuda")
            self.use_multi_gpu = config.training.distributed and torch.cuda.device_count() > 1
            if self.use_multi_gpu:
                self.logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            else:
                self.logger.info(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            self.use_multi_gpu = False
            self.logger.info("Using CPU")

        self.model.to(self.device)
        if self.use_multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def train(self):
        train_loader, val_loader = get_data_loaders(self.config)
        num_epochs = self.config.training.num_epochs
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(train_loader)
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            eeg = batch['eeg'].to(self.device)
            embedding = batch['embedding'].to(self.device)
            text = batch['text'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(eeg, embedding)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), text.view(-1))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss