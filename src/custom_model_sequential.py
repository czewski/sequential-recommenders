import torch
from torch import nn
from torch.nn.init import normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class CustomLSTM(SequentialRecommender):
    def __init__(self, config, dataset):
        super(CustomLSTM, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.lstm_size = config['lstm_size']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.w1 = nn.LSTM(self.lstm_size, self.lstm_size, bias=False) #posso customizar o tamanho
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.loss_type = config['loss_type']
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.002)
        elif isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.05)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        #item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        lstm_output, _ = self.w1(item_seq_emb)
       # lstm_output = self.dense(lstm_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(lstm_output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
