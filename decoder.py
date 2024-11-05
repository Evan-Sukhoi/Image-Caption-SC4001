from unittest.mock import sentinel
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda:0


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim, mode=None):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.mode = mode
        
        if mode == "aoa":
            self.aoa_layer = nn.Sequential(
                nn.Linear((1 + 1) * encoder_dim, 2 * encoder_dim),  # (query + context) x encoder_dim
                nn.GLU()
            )
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # [batch_size_t, num_pixels=196, 2048] -> [batch_size_t, num_pixels, attention_dim]
        att2 = self.decoder_att(decoder_hidden)  # [batch_size_t, decoder_dim=512] -> [batch_size_t, attention_dim]

        # Additive Attention (Different from LuongAttention)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # [batch_size_t, num_pixels=196, attention_dim] -> [batch_size_t, num_pixels]
        alpha = self.softmax(att)  # [batch_size_t, num_pixels=196]
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [batch_size_t, encoder_dim=2048]
        
        if self.mode == "aoa":
            # Apply AoA mechanism to combine attention and context
            aoa_input = torch.cat([attention_weighted_encoding, decoder_hidden], dim=1)
            output = self.aoa_layer(aoa_input)

        return output, alpha


    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)

        a_g = self.avgpool(out)  # (batch_size, 2048, 1, 1)
        a_g = a_g.view(a_g.size(0), -1)   # (batch_size, 2048)

        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        v_g = F.relu(self.affine_embed(a_g))

        return out, v_g

class Adaptive_Attention(Attention):
    """
    Adaptive Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Adaptive_Attention, self).__init__(encoder_dim, decoder_dim, attention_dim)
        self.sentinel_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.affine_s_t = nn.Linear(decoder_dim, encoder_dim)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden, s_t):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :param s_t: sentinel vector, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(torch.tanh(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        att_sentinel = self.full_att(torch.tanh(self.sentinel_att(s_t) + self.decoder_att(decoder_hidden)))
        att = torch.cat([att, att_sentinel], dim=1)

        alpha = self.softmax(att)  # (batch_size, num_pixels + 1)

        # c_hat_t = beta * s_t + （1-beta）* c_t
        attention_weighted_s_t = s_t * alpha[:, -1].unsqueeze(1)
        attention_weighted_s_t = self.affine_s_t(self.dropout(attention_weighted_s_t))
        attention_weighted_encoding = (encoder_out * alpha[:, :-1].unsqueeze(2)).sum(dim=1)\
                                      + attention_weighted_s_t  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class RNN_LSTM_DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5, visual_flat=196, attn_type=None):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(RNN_LSTM_DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.attn_type = attn_type

        if attn_type == "adaptive":
            self.adaptive_attention = Adaptive_Attention(self.encoder_dim, decoder_dim, attention_dim)  # attention network
            self.decode_step_adaptive = nn.LSTMCell(2 * embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
            self.affine_embed = nn.Linear(embed_dim, decoder_dim)  # linear layer to transform embeddings
            self.affine_decoder = nn.Linear(decoder_dim, decoder_dim)  # linear layer to transform decoder's output
            self.fc_encoder = nn.Linear(self.encoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        else:
            self.attention = Attention(encoder_dim, decoder_dim, attention_dim) 
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        self.fine_tune_embeddings()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
        if self.attn_type == "adaptive":
            self.fc_encoder.bias.data.fill_(0)
            self.fc_encoder.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)  # [batch_size, 196, 2048] -> [batch_size, 2048]
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        # Flatten image
        if self.attn_type == "adaptive":
            encoder_out, v_g = encoder_out
        
        # [batch_size, 14, 14, 2048]/[batch_size, 196, 2048] -> [batch_size, 196, 2048]
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1) # 2048
        vocab_size = self.vocab_size

        # Flatten image -> [batch_size, num_pixels=196, encoder_dim=2048]
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? For each of data in the batch, when len(prediction) = len(caption_lengths), Stop.
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # [batch_size, max_caption_length=52, embed_dim]

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # [batch_size, decoder_dim]

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            
            if self.attn_type == "adaptive":
                g_t = self.sigmoid(self.affine_embed(self.dropout(embeddings[:batch_size_t, t, :]))
                                   + self.affine_decoder(self.dropout(h[:batch_size_t])))    # (batch_size_t, decoder_dim)

                # s_t = g_t * tanh(c_t)
                s_t = g_t * torch.tanh(c[:batch_size_t])   # (batch_size_t, decoder_dim)

                h, c = self.decode_step_adaptive(
                    torch.cat([embeddings[:batch_size_t, t, :], v_g[:batch_size_t, :]], dim=1),
                               (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
                attention_weighted_encoding, alpha = self.adaptive_attention(encoder_out[:batch_size_t], h[:batch_size_t], s_t)

                preds = self.fc(self.dropout(h)) + self.fc_encoder(self.dropout(attention_weighted_encoding))
                predictions[:batch_size_t, t, :] = preds
                alphas[:batch_size_t, t, :] = alpha[:, :-1]
                
            else:
                # alpha: [batch_size_t, 196]
                # attention_weighted_encoding: [batch_size_t, 2048]
                attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
                gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
                # torch.cat([batch_size_t, 500], [batch_size_t, 2048], dim=1) = [batch_size_t, 2548] -> [batch_size_t, 512]
                h, c = self.lstm(
                    torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))
                preds = self.fc(self.dropout(h))  # [batch_size_t, vocab_size]
                predictions[:batch_size_t, t, :] = preds
                alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


class RNN_DecoderWithAttention(nn.Module):
    """
    RNN Decoder with Attention mechanism.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: Attention network size
        :param embed_dim: Embedding size
        :param decoder_dim: RNN decoder's hidden size
        :param vocab_size: Vocabulary size
        :param encoder_dim: Feature size of encoded images
        :param dropout: Dropout probability
        """
        super(RNN_DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # Attention layer
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim) 
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=self.dropout)
        
        # RNNCell
        self.rnn = nn.RNNCell(embed_dim + encoder_dim, decoder_dim) 
        
        # Initial hidden state
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        
        # Gate layer for attention-weighted encoding
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        # Fully connected output layer
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
        self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allows fine-tuning of embedding layer.
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Initializes the initial hidden state for RNN based on encoded images.
        """
        mean_encoder_out = encoder_out.mean(dim=1)  # [batch_size, encoder_dim]
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        return h

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # [batch_size, num_pixels, encoder_dim]
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # [batch_size, max_caption_length, embed_dim]

        # Initialize RNN hidden state
        h = self.init_hidden_state(encoder_out)

        # Define decode lengths (ignore <end>)
        decode_lengths = (caption_lengths - 1).tolist()

        # Prepare tensors for predictions and attention weights
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # Decode step by step
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            
            # Apply attention
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar
            attention_weighted_encoding = gate * attention_weighted_encoding

            # RNN step
            h = self.rnn(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                h[:batch_size_t]
            )

            # Output layer (predict next word)
            preds = self.fc(self.dropout(h))  # [batch_size_t, vocab_size]
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


import torch
import torch.nn as nn

class RNN_LSTM_DecoderWithoutAttention(nn.Module):
    """
    Decoder without attention.
    """

    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(RNN_LSTM_DecoderWithoutAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        self.fine_tune_embeddings()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)  # [batch_size, 196, 2048] -> [batch_size, 2048]
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, sort indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image -> [batch_size, num_pixels=196, encoder_dim=2048]
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # [batch_size, max_caption_length, embed_dim]

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # [batch_size, decoder_dim]

        # Decode lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensor to hold word prediction scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        # Decode step-by-step
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h, c = self.lstm(
                torch.cat([embeddings[:batch_size_t, t, :], encoder_out[:batch_size_t].mean(dim=1)], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))  # [batch_size_t, vocab_size]
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind

