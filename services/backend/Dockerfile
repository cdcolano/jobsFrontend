FROM python:3.8-slim-buster

RUN mkdir app
WORKDIR /app

ENV PATH="${PATH}:/root/.local/bin"
ENV PYTHONPATH=.

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r ./requirements.txt

COPY src/ .