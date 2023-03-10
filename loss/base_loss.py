import torch.nn as nn

class BaseLoss(nn.Module):

    """ Base loss class.
    args:
        weight: weight of current loss.
        input_keys: keys for actual inputs to calculate_loss().
            Since "inputs" may contain many different fields, we use input_keys
            to distinguish them.
        loss_func: the actual loss func to calculate loss.
    """

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.input_keys = ['input']
        self.loss_func = lambda: 0

    def calculate_loss(self, **kwargs):
        return self.loss_func(*[kwargs[key] for key in self.input_keys])    

    def forward(self, inputs):
        actual_inputs = {}
        for input_key in self.input_keys:
            actual_inputs.update({input_key: inputs[input_key]})
        return self.weight * self.calculate_loss(**actual_inputs)
