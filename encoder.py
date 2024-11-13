import torch
from torch import nn
import torch.nn.functional as F
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda:0


class CNN_Encoder(nn.Module):
    """
    CNN_Encoder.
    """

    def __init__(self, encoded_image_size=14, attention_method="ByPixel"):
        super(CNN_Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.attention_method = attention_method

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        # Specifically, Remove: AdaptiveAvgPool2d(output_size=(1, 1)), Linear(in_features=2048, out_features=1000, bias=True)]
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        if self.attention_method == "ByChannel":
            self.cnn1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.relu = nn.ReLU(inplace=True)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        self.start_fine_tune_epoch = None
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images [batch_size, encoded_image_size=14, encoded_image_size=14, 2048]
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        if self.attention_method == "ByChannel":  # [batch_size, 2048, 8, 8] -> # [batch_size, 512, 8, 8]
            out = self.relu(self.bn1(self.cnn1(out)))
        out = self.adaptive_pool(out)  # [batch_size, 2048/512, 8, 8] -> [batch_size, 2048/512, 14, 14]
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: True to allow fine-tuning, False to prevent.
        """
        
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
                
    def start_fine_tuning_if_ready(self, current_epoch):
        """
        Start fine-tuning if ready.
        :param current_epoch: Current epoch number.
        """
        if self.start_fine_tune_epoch is not None and current_epoch >= self.start_fine_tune_epoch:
            print("Fine-tuning encoder at epoch {}.".format(current_epoch))
            self.fine_tune(fine_tune=True)
        else:
            print("Not fine-tuning encoder yet at epoch {}. Will start at epoch {}.".format(current_epoch, self.start_fine_tune_epoch))


class Adaptive_Encoder(CNN_Encoder):
    """
    Adaptive_Encoder.
    """
    def __init__(self, encoded_image_size, embed_dim, decoder_dim):
        super(Adaptive_Encoder, self).__init__(encoded_image_size)
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.dropout = nn.Dropout(0.5)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.avgpool = nn.AvgPool2d(encoded_image_size)
        self.affine_embed = nn.Linear(2048, embed_dim)
        self.fine_tune()

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
