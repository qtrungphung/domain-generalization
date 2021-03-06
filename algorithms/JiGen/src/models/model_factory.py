from algorithms.JiGen.src.models import resnet
from algorithms.JiGen.src.models import caffenet

nets_map = {
    'caffenet': caffenet.caffenet,
    'resnet18': resnet.resnet18,
    'resnet50': resnet.resnet50
}

def get_model(name):
    if name not in nets_map:
        raise ValueError('Name of model unknown %s' % name)

    def get_model_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_model_fn