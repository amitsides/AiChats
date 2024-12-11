
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_rnn_data_parallel_sample(
    project: str,
    endpoint_id: str,
    gcs_source_uri: str,
    gcs_destination_output_uri_prefix: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instance = Value()
    instance.string_value = gcs_source_uri
    # See gs://google-cloud-aiplatform/schema/predict/params/rnn_1.0.0.yaml for the format of the parameters.
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    instances = [instance]
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/rnn_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))

    output_config = {
        "predictions_format": "jsonl",
        "output_uri_prefix": gcs_destination_output_uri_prefix,
    }
    gcs_destination = {"output_uri_prefix": gcs_destination_output_uri_prefix}
    output_info = client.predict(
        endpoint=endpoint,
        instances=instances,
        parameters=parameters,
        output_config=output_config,
    )
    print("output_info")
    print(" gcs_destination:", output_info.gcs_destination)

  
```