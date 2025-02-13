FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip setuptools 

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY prepare.py /app/

CMD ["python", "prepare.py"]