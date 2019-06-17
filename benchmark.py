dependencies = ['torch', 'torchvision']

from efficientnet_pytorch.model import EfficientNet
from sotabench.image_classification import imagenet

def benchmark():
    imagenet.benchmark(
        model=EfficientNet.from_pretrained(model_name='efficientnet-b0'),
        paper_model_name='EfficientNet B0',
        paper_arxiv_id='1905.11946',
        paper_pwc_id='efficientnet-rethinking-model-scaling-for'
    )
    
    imagenet.benchmark(
        model=EfficientNet.from_pretrained(model_name='efficientnet-b1'),
        paper_model_name='EfficientNet B1',
        paper_arxiv_id='1905.11946',
        paper_pwc_id='efficientnet-rethinking-model-scaling-for'
    )
    
    imagenet.benchmark(
        model=EfficientNet.from_pretrained(model_name='efficientnet-b2'),
        paper_model_name='EfficientNet B2',
        paper_arxiv_id='1905.11946',
        paper_pwc_id='efficientnet-rethinking-model-scaling-for'
    )
    
    imagenet.benchmark(
        model=EfficientNet.from_pretrained(model_name='efficientnet-b3'),
        paper_model_name='EfficientNet B3',
        paper_arxiv_id='1905.11946',
        paper_pwc_id='efficientnet-rethinking-model-scaling-for'
    )
