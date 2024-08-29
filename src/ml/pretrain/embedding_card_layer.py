"""
@project: THPMaster
@File   : embedding_card_layer.py
@Desc   :
@Author : gql
@Date   : 2024/8/22 16:27
"""
import torch
from torch import nn


class CardEmbeddingModel(nn.Module):
    def __init__(self, num_cards=52, embedding_dim=16, num_hand_types=10):
        super().__init__()
        self.embedding = nn.Embedding(num_cards, embedding_dim)
        self.hand_type_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 7, 1024),
            nn.Linear(1024, num_hand_types),
        )
        self.win_rate_regressor = nn.Sequential(
            nn.Linear(embedding_dim * 7, 1024),
            nn.Linear(1024, 1),
        )
        self.potential_hand_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 7, 1024),
            nn.Linear(1024, 1)
        )

    def forward(self, hand_indices, public_indices):
        hand_embeddings = self.embedding(hand_indices)
        public_embeddings = self.embedding(public_indices)
        combined_embedding = torch.cat([hand_embeddings.view(hand_embeddings.size(0), -1),
                                        public_embeddings.view(public_embeddings.size(0), -1)], dim=1)
        hand_type_logits = self.hand_type_classifier(combined_embedding)
        win_rate = self.win_rate_regressor(combined_embedding)
        potential_prob = self.potential_hand_predictor(combined_embedding)

        return hand_type_logits, win_rate, potential_prob


def compute_loss(hand_type_logits, true_hand_type, win_rate, true_win_rate, potential_prob, true_potential_prob):
    loss_hand_type = nn.CrossEntropyLoss()(hand_type_logits, true_hand_type)
    loss_win_rate = nn.MSELoss()(win_rate, true_win_rate)
    loss_potential_prob = nn.BCELoss()(potential_prob, true_potential_prob)
    return loss_hand_type + loss_win_rate + loss_potential_prob


def train_card_embedding_model(model, dataloader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (hand_indices, board_indices, true_hand_type, true_win_rate, true_potential_prob) in enumerate(
                dataloader):
            optimizer.zero_grad()
            hand_type_logits, win_rate, potential_prob = model(hand_indices, board_indices)
            loss = compute_loss(hand_type_logits, true_hand_type, win_rate, true_win_rate,
                                potential_prob, true_potential_prob)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:  # 每 10 个批次打印一次
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0


if __name__ == '__main__':
    pass
