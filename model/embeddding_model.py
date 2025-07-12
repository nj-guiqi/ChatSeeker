import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from transformers import AutoTokenizer, AutoModel

class DenseRetrievalModel(nn.Module):
    def __init__(self, config):
        super(DenseRetrievalModel, self).__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(config['model_path'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
        self.max_length = config['max_length']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embedding(self, query):
        query_enc = self.tokenizer(query, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
        input_ids = query_enc["input_ids"].to(self.device)
        attention_mask = query_enc["attention_mask"].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # CLS embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # 形状: (1, hidden_size)
        cls_embedding = F.normalize(cls_embedding, p=2, dim=-1)
        cls_embedding = cls_embedding.squeeze(0)  # 形状: (hidden_size,)

        # embedding -> np.array
        cls_embedding = cls_embedding.cpu().detach().numpy()
        return cls_embedding