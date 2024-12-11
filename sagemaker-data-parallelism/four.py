import sagemaker
from sagemaker.huggingface import HuggingFace

# Set up SageMaker session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Define hyperparameters
hyperparameters = {
    'epochs': 3,
    'per_device_train_batch_size': 32,
    'model_name_or_path': 'bert-base-uncased',
    'output_dir': '/opt/ml/model'
}

# Create HuggingFace estimator
huggingface_estimator = HuggingFace(
    entry_point='train.py',
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    role=role,
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39',
    hyperparameters=hyperparameters
)

# Start training
huggingface_estimator.fit({'train': 's3://your-bucket/train-data', 'test': 's3://your-bucket/test-data'})

# Deploy the model
predictor = huggingface_estimator.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge')

# Make a prediction
data = {"inputs": "Your input text here"}
prediction = predictor.predict(data)