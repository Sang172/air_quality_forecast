import boto3
import json
import os
import time

def lambda_handler(event, context):
    """Triggers a SageMaker training job based on a previous job's configuration.

    Input Type: event (dict), context (LambdaContext)
    Output Type: dict - {'statusCode': int, 'body': str}
    """
    sagemaker = boto3.client('sagemaker')
    base_job_name = os.environ.get('BASE_TRAINING_JOB_NAME')
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    training_job_name = f"{base_job_name}-{timestamp}"

    try:
        response = sagemaker.describe_training_job(TrainingJobName=base_job_name)

        response.pop('TrainingJobArn', None)
        response.pop('TrainingJobStatus', None)
        response.pop('SecondaryStatus', None)
        response.pop('CreationTime', None)
        response.pop('TrainingStartTime', None)
        response.pop('TrainingEndTime', None)
        response.pop('LastModifiedTime', None)
        response.pop('ResponseMetadata', None)
        response.pop('ModelArtifacts', None)
        response.pop('SecondaryStatusTransitions', None)
        response.pop('TrainingTimeInSeconds', None)
        response.pop('BillableTimeInSeconds', None)
        response.pop('ProfilingStatus', None)

        response['TrainingJobName'] = training_job_name
        create_response = sagemaker.create_training_job(**response)

        print(f"Created SageMaker training job: {create_response['TrainingJobArn']}")
        return {
            'statusCode': 200,
            'body': json.dumps(f"Created SageMaker training job: {create_response['TrainingJobArn']}")
        }
    except Exception as e:
        print(f"Error creating training job: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error creating training job: {e}")
        }