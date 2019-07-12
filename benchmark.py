from efficientnet_pytorch.model import EfficientNet
from sotabench.image_classification import ImageNet
import torchvision.transforms as transforms
import PIL

# Define Transforms    
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
input_transform = transforms.Compose([
    transforms.Resize(256, PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Run Evaluation
ImageNet.benchmark(
    model=EfficientNet.from_pretrained(model_name='efficientnet-b0'),
    paper_model_name='EfficientNet-B0',
    paper_arxiv_id='1905.11946',
    paper_pwc_id='efficientnet-rethinking-model-scaling-for',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)

# Run Evaluation
ImageNet.benchmark(
    model=EfficientNet.from_pretrained(model_name='efficientnet-b1'),
    paper_model_name='EfficientNet-B1',
    paper_arxiv_id='1905.11946',
    paper_pwc_id='efficientnet-rethinking-model-scaling-for',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)

# Run Evaluation
ImageNet.benchmark(
    model=EfficientNet.from_pretrained(model_name='efficientnet-b2'),
    paper_model_name='EfficientNet-B2',
    paper_arxiv_id='1905.11946',
    paper_pwc_id='efficientnet-rethinking-model-scaling-for',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)

# Run Evaluation
ImageNet.benchmark(
    model=EfficientNet.from_pretrained(model_name='efficientnet-b3'),
    paper_model_name='EfficientNet-B3',
    paper_arxiv_id='1905.11946',
    paper_pwc_id='efficientnet-rethinking-model-scaling-for',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)
