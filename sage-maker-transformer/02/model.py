import sagemaker
from sagemaker.huggingface import HuggingFace

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

hyperparameters = {
    "model_name": "improved-transformer",
    "b": 32,
    "s": 128,
    "h": 512,
    "n": 8,
    "v": 30522,
    "p": 4,
    "q": 64,
    "N": 6,
    "dropout": 0.1,
    "epochs": 10,
    "learning_rate": 0.001,
    "output_format": "safetensors"
}

ecr_image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04"

huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir="./source_dir",
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.17.0",
    pytorch_version="1.10.2",
    py_version="py38",
    hyperparameters=hyperparameters,
    image_uri=ecr_image
)

huggingface_estimator.fit()