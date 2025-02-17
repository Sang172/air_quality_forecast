FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.12.0-gpu-py310-cu118-ubuntu20.04

WORKDIR /opt/ml/code

COPY requirements.txt /opt/ml/code/
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py /opt/ml/code/

ENV SAGEMAKER_PROGRAM train.py