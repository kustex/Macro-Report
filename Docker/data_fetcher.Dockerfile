FROM python:3.10.12-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip &&\
    pip install -r requirements.txt

CMD ["python", "src/data_fetcher.py"]
