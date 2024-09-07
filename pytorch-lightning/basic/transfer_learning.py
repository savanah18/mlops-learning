from lightning import LightningModule

from modules import Encoder, Decoder


class BaseAutoEncoder(LightningModule):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

# create a ligtning module CIFAR10 classifier, use BaseAutoEncoder as autoencoder
class CIFAR10Classifier(LightningModule):
    def __init__(self) -> None:
        # init the pretaineed lightning module
        self.feature_extractor = BaseAutoEncoder()

# TODO: Implement the CIFAR10Classifier class
# TODO: Implement ImagenetTransferLearning class
