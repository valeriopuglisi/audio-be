#FROM python:3.10-slim-buster
FROM nvcr.io/nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && apt-get clean

WORKDIR /app
COPY pip-requirements.txt /app/pip-requirements.txt
RUN pip install -r pip-requirements.txt
COPY . /app
RUN mkdir -p /app/media
EXPOSE 5000

CMD ["/usr/bin/python3", "main.py"]
