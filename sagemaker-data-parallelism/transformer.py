import sagemaker
from sagemaker.pytorch import PyTorch

# Define the training script
training_script = "train.py"

# Define the S3 bucket and input/output paths
bucket = "your-bucket-name"
input_data_path = f"s3://{bucket}/input_data"
output_path = f"s3://{bucket}/output_data"

# Define the training parameters
hyperparameters = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 10,
}

# Create a PyTorch estimator
estimator = PyTorch(
    entry_point=training_script,
    role="your-sagemaker-role",
    instance_count=1,
    instance_type="ml.p3.2xlarge",
    framework_version="1.10.1",
    py_version="py38",
    hyperparameters=hyperparameters,
    output_path=output_path,
)

# Define the input data configuration
data_channels = {"training": input_data_path}

# Train the model
estimator.fit(data_channels)

# Deploy the model (optional)
# endpoint_name = "my-transformer-endpoint"
# predictor = estimator.deploy(initial_instance_count=1, instance_type="ml.m5.large", endpoint_name=endpoint_name)