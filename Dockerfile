FROM python:3.9.6-slim-buster

RUN apt-get update && apt-get install -y conda && apt-get clean

WORKDIR /app
COPY . /app

RUN conda install -r requirements.txt

EXPOSE 5000

CMD ["python", "main.py"]
