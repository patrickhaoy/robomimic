"""
A simple example showing how to add custom observation modalities, and custom
observation networks (EncoderCore, ObservationRandomizer, etc.) as well.
We also show how to use your custom classes directly in a config, and link them to
your environment's observations
"""

import numpy as np
import torch
import robomimic
from robomimic.models import EncoderCore, Randomizer
from robomimic.utils.obs_utils import Modality, ScanModality
from robomimic.config.base_config import BaseConfig


# Let's create a new modality to handle observation modalities, which will be interpreted as
# single frame images, with raw shape (H, W) in range [0, 255]
class CustomImageModality(Modality):
    # We must define the class string name to reference this modality with the @name attribute
    name = "custom_image"

    # We must define two class methods: a processor and an unprocessor method. The processor
    # method should map the raw observations (a numpy array OR torch tensor) into a form / shape suitable for learning,
    # and the unprocess method should do the inverse operation
    @classmethod
    def _default_obs_processor(cls, obs):
        # We add a channel dimension and normalize them to be in range [-1, 1]
        return (obs / 255.0 - 0.5) * 2

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        # We do the reverse
        return ((obs / 2) + 0.5) * 255.0


# You can also modify pre-existing modalities as well. Let's say you have scan data that pads the ends with a 0, so we
# want to pre-process those scans in a different way. We can specify a custom processor / unprocessor
# method that will override the default one (assumes obs are a flat 1D array):
def custom_scan_processor(obs):
    # Trim the padded ends
    return obs[1:-1]


def custom_scan_unprocessor(obs):
    # Re-add the padding
    # Note: need to check type
    return np.concatenate([np.zeros(1), obs, np.zeros(1)]) if isinstance(obs, np.ndarray) else \
        torch.concat([torch.zeros(1), obs, torch.zeros(1)])


# Override the default functions for ScanModality
ScanModality.set_obs_processor(processor=custom_scan_processor)
ScanModality.set_obs_unprocessor(unprocessor=custom_scan_unprocessor)


# Let's now create a custom encoding class for the custom image modality
class CustomImageEncoderCore(EncoderCore):
    # For simplicity, this will be a pass-through with some simple kwargs
    def __init__(
            self,
            input_shape,        # Required, will be inferred automatically at runtime

            # Any args below here you can specify arbitrarily
            welcome_str,
    ):
        # Always need to run super init first and pass in input_shape
        super().__init__(input_shape=input_shape)

        # Anything else should can be custom to your class
        # Let's print out the welcome string
        print(f"Welcome! {welcome_str}")

    # We need to always specify the output shape from this model, based on a given input_shape
    def output_shape(self, input_shape=None):
        # this is just a pass-through, so we return input_shape
        return input_shape

    # we also need to specify the forward pass for this network
    def forward(self, inputs):
        # just a pass through again
        return inputs


# Let's also create a custom randomizer class for randomizing our observations
class CustomImageEncoderCore(EncoderCore):
    # For simplicity, this will be a pass-through with some simple kwargs
    def __init__(
            self,
            input_shape,        # Required, will be inferred automatically at runtime

            # Any args below here you can specify arbitrarily
            welcome_str,
    ):
        # Always need to run super init first and pass in input_shape
        super().__init__(input_shape=input_shape)

        # Anything else should can be custom to your class
        # Let's print out the welcome string
        print(f"Welcome! {welcome_str}")

    # We need to always specify the output shape from this model, based on a given input_shape
    def output_shape(self, input_shape=None):
        # this is just a pass-through, so we return input_shape
        return input_shape

    # we also need to specify the forward pass for this network
    def forward(self, inputs):
        # just a pass through again
        return inputs

class CustomImageRandomizer(Randomizer):
    # TODO: Ajay: Can you add an example of a custom Randomizer here? Read through the forward_in / forward_out but still confused as to what they do
    pass


# Now, we can directly reference the classes in our config!
config = BaseConfig()
config.observation.encoder.custom_image.core_class = "CustomImageEncoderCore"       # Custom class, in string form
config.observation.encoder.custom_image.core_kwargs.welcome_str = "hi there!"       # Any custom arguments, of any primitive type that is json-able
config.observation.encoder.custom_image.obs_randomizer_class = "CustomImageRandomizer"
config.observation.encoder.custom_image.obs_randomizer_kwargs.XXX = "todo"      # todo Ajay
