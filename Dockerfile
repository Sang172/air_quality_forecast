FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install waitress

COPY . .

EXPOSE 5000

CMD exec gunicorn --bind :${PORT} --log-level debug app:app