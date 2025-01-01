import snowflake.connector
import pandas as pd
from sagemaker import Session
from sagemaker.feature_store.feature_group import FeatureGroup
import boto3

def setup_snowflake_connection(account, user, password, warehouse, database, schema):
    """
    Establish connection to Snowflake
    """
    conn = snowflake.connector.connect(
        account=account,
        user=user,
        password=password,
        warehouse=warehouse,
        database=database,
        schema=schema
    )
    return conn

def create_feature_store(session, feature_group_name, record_identifier_name):
    """
    Create SageMaker Feature Store
    """
    feature_group = FeatureGroup(
        name=feature_group_name,
        feature_definitions=None,  # Will be set when data is loaded
        sagemaker_session=session
    )
    
    feature_group.create(
        s3_uri=f"s3://{bucket_name}/{feature_group_name}",
        record_identifier_name=record_identifier_name,
        enable_online_store=True,
        role_arn=role_arn
    )
    return feature_group

def load_data_from_snowflake_to_sagemaker(conn, query, feature_group):
    """
    Extract data from Snowflake and load into SageMaker Feature Store
    """
    # Execute Snowflake query
    cur = conn.cursor()
    cur.execute(query)
    
    # Convert to DataFrame
    df = cur.fetch_pandas_all()
    
    # Load into Feature Store
    feature_group.ingest(
        data_frame=df,
        max_workers=3,
        wait=True
    )
    return df

def setup_sagemaker_processing_job(role, instance_type='ml.m5.xlarge'):
    """
    Configure SageMaker processing job for data preparation
    """
    processor = SKLearnProcessor(
        framework_version='0.23-1',
        role=role,
        instance_type=instance_type,
        instance_count=1,
        base_job_name='snowflake-preprocessing'
    )
    return processor

def create_training_pipeline():
    """
    Set up SageMaker training pipeline with Snowflake data
    """
    pipeline = Pipeline(
        name="SnowflakeTrainingPipeline",
        parameters=[
            ProcessingInput(
                source=feature_store_s3_uri,
                destination="/opt/ml/processing/input"
            )
        ],
        steps=[
            ProcessingStep(
                name="PreprocessData",
                processor=processor,
                inputs=[processing_input],
                outputs=[ProcessingOutput(output_name="train", source="/opt/ml/processing/train")],
                code="preprocess.py"
            ),
            TrainingStep(
                name="TrainModel",
                estimator=estimator,
                inputs={
                    "training": TrainingInput(
                        s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                    )
                }
            )
        ]
    )
    return pipeline

# Best practices for configuration
def get_optimal_configuration(data_size, model_complexity):
    """
    Return optimal configuration based on data size and model complexity
    """
    configs = {
        'small': {
            'instance_type': 'ml.m5.xlarge',
            'batch_size': 128,
            'max_parallel_processes': 2
        },
        'medium': {
            'instance_type': 'ml.m5.2xlarge',
            'batch_size': 256,
            'max_parallel_processes': 4
        },
        'large': {
            'instance_type': 'ml.m5.4xlarge',
            'batch_size': 512,
            'max_parallel_processes': 8
        }
    }
    
    # Logic to determine configuration based on input parameters
    if data_size < 10_000_000:  # 10M records
        return configs['small']
    elif data_size < 100_000_000:  # 100M records
        return configs['medium']
    else:
        return configs['large']