import requests

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import scipy
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image

from caption_models import PseudoInception, CaptionNet
from caption_models import BeheadedInception3

DEFAULT_BOS_IDX = 1
DEFAULT_EOS_IDX = 2


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = {'id': id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = {'id' : id, 'confirm': token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


def load_models(inception_path, caption_net_path, vocab_size, padding_idx,
                transform_input = True,  pseudo_inception = False):
    """
    returns two parts of a model and an available device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    caption_model = CaptionNet(vocab_size = vocab_size, padding_idx = padding_idx)
    caption_model.load_state_dict(torch.load(caption_net_path, map_location=torch.device('cpu')))
    caption_model = caption_model.to(device)
    caption_model.eval()

    if pseudo_inception:
        inception = PseudoInception()
    else:
        inception = BeheadedInception3(transform_input=transform_input)
        inception.load_state_dict(torch.load(inception_path, map_location=torch.device('cpu')))
    inception = inception.to(device)
    inception.eval()

    return inception, caption_model, device


def image_load(img_path):
    img = plt.imread(img_path)
    return img

def image_process(img):
    img_size_init = img.shape[:2]
    img_for_net = resize(img, (299, 299))#.astype('float32') / 255.  ### this is done by resize
    return img, img_for_net, img_size_init

def image_show(img, size = (500, 500), figsize = (8, 8), show = False):
    img_show = resize(img, size)
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    ax.set_axis_off()
    _ = ax.imshow(img_show)
    if not show:
        plt.ioff()
        plt.close()
    return fig

def generate_caption(image, inception, caption_model, idx2word,
                     exclude_from_prediction = None,
                     caption_prefix=(DEFAULT_BOS_IDX,), end_token_idx = DEFAULT_EOS_IDX, 
                     temperature = 1, sample = True, max_len = 10):
    """
    generates a caption on normalized image in form of np.array
    """

    assert isinstance(image, np.ndarray) and np.max(image) <= 1\
           and np.min(image) >=0 and image.shape[-1] == 3
    
    if not exclude_from_prediction:
        exclude_from_prediction = []

    with torch.no_grad():
        image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)

        vectors_8x8, vectors_neck, logits = inception(image[None]) #### adding batch size, then channels, then h, w
        
        caption_prefix = list(caption_prefix)
        text_caption = []

        with torch.no_grad():
        # cлово за словом генерируем описание картинки
        # actually what is happening is that every step i append last predicted word but run the net from the beginning with the same init vec
          for _ in range(max_len):
              # 1. Представляем caption_prefix в виде матрицы
              # 2. Получаем из RNN-ки логиты, передав ей vectors_neck и матрицу из п.1
              # 3. Переводим логиты RNN-ки в вероятности (например, с помощью F.softmax)
              # 4. Сэмплируем следующее слово в описании, используя полученные вероятности. Можно сэмплировать жадно, можно сэмплировать из распределения
              # 5. Добавляем новое слово в caption_prefix
              # 6. Если RNN-ка сгенерила символ конца предложения, останавливаемся
            
            captions_ix_inp = torch.tensor(caption_prefix, dtype = torch.long).unsqueeze(0)
            logits_for_next = caption_model.forward(vectors_neck, captions_ix_inp)
            next_token_distr = F.softmax(logits_for_next[0, -1, :] / temperature, dim = -1).numpy() ### fetching only last prediction
            next_token_idx_hard = next_token_distr.argmax(axis = -1)

            # fixing prediction due to possibility of spec tokens
            zeroing_mask = np.zeros(next_token_distr.shape[0], dtype=bool)
            zeroing_mask[exclude_from_prediction] = True
            relocate_proba_sum = np.sum(next_token_distr[zeroing_mask])
            next_token_distr[zeroing_mask] = 0  #### zeroing out spec tokens
            next_token_distr[~zeroing_mask] += relocate_proba_sum / np.sum(~zeroing_mask) #### renormalizing probabilities
            # next_token_distr = scipy.special.softmax(next_token_distr) #### renormalizing probabilities

            next_token_idx_sampled = np.random.choice(np.arange(len(next_token_distr)), p = next_token_distr)
            
            if sample:
                if next_token_idx_sampled == end_token_idx:
                    break
                text_caption.append(idx2word[next_token_idx_sampled])
                caption_prefix.append(next_token_idx_sampled)
            else:
                if next_token_idx_hard == end_token_idx:
                    break
                text_caption.append(idx2word[next_token_idx_hard])
                caption_prefix.append(next_token_idx_hard)
        
    return ' '.join(text_caption)


def get_captions(img, inception, caption_model, idx2word,
                 exclude_from_prediction = None,
                 caption_prefix = (DEFAULT_BOS_IDX,), end_token_idx = DEFAULT_EOS_IDX,
                 n_captions = 1, temperature = 1, sample = True, max_len = 10
                 ):

    if not exclude_from_prediction:
        exclude_from_prediction = []
        
    captions = []
    for _ in range(n_captions):
        generated_caption = generate_caption(img, inception, caption_model, idx2word,
                                             exclude_from_prediction = exclude_from_prediction,
                                             caption_prefix = caption_prefix, end_token_idx = end_token_idx, 
                                             temperature = temperature, sample = sample, max_len = max_len)
        captions.append(generated_caption)
    return captions