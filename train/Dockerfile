FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.12.0-gpu-py310

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

COPY requirements.txt /opt/ml/code/
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py /opt/ml/code/

ENV SM_MODEL_DIR /opt/ml/model
ENV SM_OUTPUT_DATA_DIR /opt/ml/output/data
ENV SM_CHANNEL_TRAINING /opt/ml/input/data/training
ENV SM_NUM_GPUS 1

CMD ["python", "train.py"]