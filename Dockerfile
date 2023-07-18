FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y python3-pip && apt-get clean

WORKDIR /app
COPY pip-requirements.txt /app/pip-requirements.txt
RUN pip install -r pip-requirements.txt
COPY . /app

EXPOSE 5000

CMD ["python", "main.py"]
