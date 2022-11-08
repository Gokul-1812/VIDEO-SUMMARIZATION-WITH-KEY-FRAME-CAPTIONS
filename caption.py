import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from matplotlib.pyplot import imread
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def okei(s):
  s.pop(0)
  s.pop(-1)
  st=''
  for i in range(len(s)):
    if i==0:
      st=st+s[i].capitalize()
    else:
      st=st+' '+s[i]
  
  return st

def caption(source_vid):
    source_vid=source_vid[:-4]
    model='/content/drive/MyDrive/VSWKC/ImageCaptioning-MSCOCO/checkpoint.pth.tar'
    # Load model
    checkpoint = torch.load(model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    word_map='/content/drive/MyDrive/VSWKC/ImageCaptioning-MSCOCO/WORDMAP.json'
    # Load word map (word2ix)
    with open(word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    beam_size=3
    
    # Encode, decode with attention and beam search
    def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
      k = beam_size
      vocab_size = len(word_map)

      # Read image and process
      img = imread(image_path)
      if len(img.shape) == 2:
          img = img[:, :, np.newaxis]
          img = np.concatenate([img, img, img], axis=2)
      img = img.transpose(2, 0, 1)
      img = img / 255.
      img = torch.FloatTensor(img).to(device)
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
      transform = transforms.Compose([normalize])
      image = transform(img)  # (3, 256, 256)

      # Encode
      image = image.unsqueeze(0)  # (1, 3, 256, 256)
      encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
      enc_image_size = encoder_out.size(1)
      encoder_dim = encoder_out.size(3)

      # Flatten encoding
      encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
      num_pixels = encoder_out.size(1)

      # We'll treat the problem as having a batch size of k
      encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

      # Tensor to store top k previous words at each step; now they're just <start>
      k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

      # Tensor to store top k sequences; now they're just <start>
      seqs = k_prev_words  # (k, 1)

      # Tensor to store top k sequences' scores; now they're just 0
      top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

      # Tensor to store top k sequences' alphas; now they're just 1s
      seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

      # Lists to store completed sequences, their alphas and scores
      complete_seqs = list()
      complete_seqs_alpha = list()
      complete_seqs_scores = list()

      # Start decoding
      step = 1
      h, c = decoder.init_hidden_state(encoder_out)

      # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
      while True:

          embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

          awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

          alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

          gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
          awe = gate * awe

          h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

          scores = decoder.fc(h)  # (s, vocab_size)
          scores = F.log_softmax(scores, dim=1)

          # Add
          scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

          # For the first step, all k points will have the same scores (since same k previous words, h, c)
          if step == 1:
              top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
          else:
              # Unroll and find top scores, and their unrolled indices
              top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

          # Convert unrolled indices to actual indices of scores
          prev_word_inds = top_k_words / vocab_size  # (s)
          next_word_inds = top_k_words % vocab_size  # (s)

          # Add new words to sequences, alphas
          seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
          seqs_alpha = torch.cat([seqs_alpha[prev_word_inds.long()], alpha[prev_word_inds.long()].unsqueeze(1)],
                                dim=1)  # (s, step+1, enc_image_size, enc_image_size)

          # Which sequences are incomplete (didn't reach <end>)?
          incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                            next_word != word_map['<end>']]
          complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

          # Set aside complete sequences
          if len(complete_inds) > 0:
              complete_seqs.extend(seqs[complete_inds].tolist())
              complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
              complete_seqs_scores.extend(top_k_scores[complete_inds])
          k -= len(complete_inds)  # reduce beam length accordingly

          # Proceed with incomplete sequences
          if k == 0:
              break
          seqs = seqs[incomplete_inds]
          seqs_alpha = seqs_alpha[incomplete_inds]
          h = h[prev_word_inds[incomplete_inds].long()]
          c = c[prev_word_inds[incomplete_inds].long()]
          encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
          top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
          k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

          # Break if things have been going on too long
          if step > 50:
              break
          step += 1

      i = complete_seqs_scores.index(max(complete_seqs_scores))
      seq = complete_seqs[i]
      alphas = complete_seqs_alpha[i]
      words = [rev_word_map[ind] for ind in seq]
      result=' '.join(words).rsplit(' ', 1)
      print(words)
      return result
    images_path="/content/drive/MyDrive/VSWKC/Summariser/static/clustered_images"+"/"+source_vid
    files=os.listdir(images_path)
    filename=[]
    caption=[]
    for i in range(0,len(files)):
      img=images_path+"/"+files[i]
      filename.append(img)
      result=caption_image_beam_search(encoder, decoder, img, word_map, beam_size)
      #result=okei(result)
      caption.append(result)
    list_of_tuples = list(zip(filename, caption)) 
    df = pd.DataFrame(list_of_tuples, columns = ['Filename', 'Caption']) 
    print(df)
    path=images_path+"/file.csv"
    df.to_csv(path)
    return "file.csv generated"

