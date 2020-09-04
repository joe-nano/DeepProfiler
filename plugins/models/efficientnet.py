import numpy
import keras
import efficientnet.keras as efn
from keras import Input, Model
from keras.layers import Dense
from keras.optimizers import Adam

from deepprofiler.learning.model import DeepProfilerModel

class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator, is_training=True)
        self.feature_model, self.optimizer, self.loss = self.define_model(config, dset)


    ## Define supported models
    def get_supported_models(self):
        return {
            0: efn.EfficientNetB0,
            1: efn.EfficientNetB1,
            2: efn.EfficientNetB2,
            3: efn.EfficientNetB3,
            4: efn.EfficientNetB4,
            5: efn.EfficientNetB5,
            6: efn.EfficientNetB6,
            7: efn.EfficientNetB7,
        }

    def get_model(self, config, input_image=None, weights=None, classes=None):
        supported_models = self.get_supported_models()
        SM = "EfficientNet supported models: " + ",".join([str(x) for x in supported_models.keys()])
        num_layers = config["train"]["model"]["params"]["conv_blocks"]
        error_msg = str(num_layers) + " conv_blocks not in " + SM
        assert num_layers in supported_models.keys(), error_msg

        model = supported_models[num_layers](input_tensor=input_image, include_top=False, weights=weights, pooling='avg')
        return model

    def define_model(self, config, dset):
        supported_models = self.get_supported_models()
        SM = "EfficientNet supported models: " + ",".join([str(x) for x in supported_models.keys()])
        num_layers = config["train"]["model"]["params"]["conv_blocks"]
        error_msg = str(num_layers) + " conv_blocks not in " + SM
        assert num_layers in supported_models.keys(), error_msg
        # Set session
        if config["profile"]["use_pretrained_input_size"]:
            input_tensor = Input((config["profile"]["use_pretrained_input_size"], config["profile"]["use_pretrained_input_size"], 3), name="input")
            model = supported_models[num_layers](input_tensor=input_tensor, include_top=True, weights='imagenet', pooling='avg')
            model.summary()
        else:
            input_shape = (
                config["dataset"]["locations"]["box_size"],  # height
                config["dataset"]["locations"]["box_size"],  # width
                len(config["dataset"]["images"][
                        "channels"])  # channels
            )
            input_image = keras.layers.Input(input_shape)
            model = self.get_model(config, input_image=input_image)
            features = keras.layers.GlobalAveragePooling2D(name="pool5")(model.layers[-1].output)

            # 2. Create an output embedding for each target
            class_outputs = []

            i = 0
            for t in dset.targets:
                y = keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(features)
                class_outputs.append(y)
                i += 1

            # 4. Create and compile model
            model = keras.models.Model(inputs=input_image, outputs=class_outputs)

        # 3. Define the loss function
        loss_func = "categorical_crossentropy"

        ## Added weight decay following tricks reported in:
        ## https://github.com/keras-team/keras/issues/2717
        regularizer = keras.regularizers.l2(0.00001)
        for layer in model.layers:
            if hasattr(layer, "kernel_regularizer"):
                setattr(layer, "kernel_regularizer", regularizer)
        model = keras.models.model_from_json(model.to_json())

        optimizer = keras.optimizers.SGD(lr=config["train"]["model"]["params"]["learning_rate"], momentum=0.9,
                                         nesterov=True)

        return model, optimizer, loss_func
