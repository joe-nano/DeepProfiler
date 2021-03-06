import keras
import keras.applications
from plugins.models import resnet 

##################################################
# DenseNet architecture as in "Densely Connected 
# Convolutional Networks" by Gao Huang, Zhuang Liu, 
# Laurens van der Maaten, Kilian Q. Weinberger
# https://arxiv.org/pdf/1608.06993.pdf
##################################################


class ModelClass(resnet.ModelClass):
    def __init__(self, config, dset, generator, val_generator):
        super().__init__(config, dset, generator, val_generator)
        self.feature_model, self.optimizer, self.loss = super().define_model(config, dset)


    def get_supported_models(self):
        return {
            121: keras.applications.DenseNet121,
            169: keras.applications.DenseNet169,
            201: keras.applications.DenseNet201
        }


