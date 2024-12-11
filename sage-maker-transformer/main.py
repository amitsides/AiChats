import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import TrainingInput
from sagemaker.outputs import TrainingOutput

# Setup SageMaker session and IAM role
session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Define S3 paths
train_data_path = "s3://my-bucket/train-data"
output_path = "s3://my-bucket/output"

# Configure training data input
train_data = TrainingInput(
    s3_data=train_data_path,
    content_type="text/csv"
)

# Define hyperparameters
hyperparameters = {
    "model_name": "bert-base-uncased",  # or your preferred model
    "rnn_type": "lstm",
    "hidden_size": 512,
    "num_layers": 2,
    "dropout": 0.2,
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001,
    "output_format": "safetensors"
}

# Create HuggingFace estimator
huggingface_estimator = HuggingFace(
    entry_point='train.py',  # Your training script
    source_dir='scripts',    # Directory containing your scripts
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    role=role,
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39',
    hyperparameters=hyperparameters,
    output_path=output_path
)

# Start training job
huggingface_estimator.fit({'train': train_data})