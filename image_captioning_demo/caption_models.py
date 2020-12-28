import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.inception import Inception3
from warnings import warn
import numpy as np

class CaptionNet(nn.Module):
    def __init__(self, vocab_size, emb_dim = 128, rnn_units = 256, n_layers = 1,
                 cnn_feature_size = 2048, padding_idx = None):
        super(self.__class__, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.rnn_units = rnn_units
        self.n_layers = n_layers
        self.cnn_feature_size = cnn_feature_size
        self.padding_idx = padding_idx
        
        # два линейных слоя, которые будут из векторов, полученных на выходе Inseption, 
        # получать начальные состояния h0 и c0 LSTM-ки, которую далее будем 
        # разворачивать во времени и генерить ею текст
        self.cnn_to_h0 = nn.Linear(self.cnn_feature_size, self.rnn_units)
        self.cnn_to_c0 = nn.Linear(self.cnn_feature_size, self.rnn_units)
        
        # вот теперь recurrent part
        # embedding for input tokens
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx = self.padding_idx)   
            
        # стакаем LSTM-слои (1 или более)
        self.rnn = nn.LSTM(self.emb_dim, self.rnn_units, self.n_layers,  batch_first=True) #(lstm embd, hid, layers)
            
        # линейный слой для получения логитов
        self.out = nn.Linear(self.rnn_units, self.vocab_size)

    def forward(self, image_vectors, captions_ix):
        """ 
        Apply the network in training mode. 
        :param image_vectors: torch tensor, содержаший выходы inсeption. Те, из которых будем генерить текст
                shape: [batch, cnn_feature_size]
        :param captions_ix: 
                таргет описания картинок в виде матрицы  [batch, caption_length]
        :returns: логиты для сгенерированного текста описания, shape: [batch, word_i, vocab_size]
        """
        initial_cell = self.cnn_to_c0(image_vectors)
        initial_hid = self.cnn_to_h0(image_vectors)

        # применим LSTM:
        # 1. инициализируем lstm state с помощью initial_* (сверху)
        # 2. скормим LSTM captions_emb
        # 3. посчитаем логиты из выхода LSTM

        captions_emb = self.embedding(captions_ix) # [batch, caption_length, emb_dim]

        ### ! this might be different when num_layers > 1
        rnn_output, (hidden, cell) = self.rnn(captions_emb, (initial_cell.unsqueeze(0), initial_hid.unsqueeze(0))) # shape: [batch, caption_length, lstm_units]
        ### RETROSPECTIVE NOTE: here above i actually changed initial_cell and initial_hid ordering on accident
        ### but they have the same dim and projected from the same vector, and it feed the whole sequence here, so i do not mix them in every single step === it doesn't actually matter
        ### below when working with attention it is fixed
        logits = self.out(rnn_output)
        
        return logits        

# class BeheadedInception3(nn.Module):
class BeheadedInception3(Inception3):
    """ Like torchvision.models.inception.Inception3 but the head goes separately """
    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        else:
            warn("Input isn't transformed")
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x_for_attn = x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x_for_capt = x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        return x_for_attn, x_for_capt, x
        
class PseudoInception(nn.Module):
    """ Like torchvision.models.inception.Inception3 but fake))))"""
    def forward(self, x):
        return None, torch.tensor(np.random.random(size = (1, 2048)),dtype=torch.float32), None