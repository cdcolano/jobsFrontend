FROM python:3.8-slim-buster

RUN mkdir app
WORKDIR /app

ENV PATH="${PATH}:/root/.local/bin"
ENV PYTHONPATH=.

COPY requirements.txt ./

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r ./requirements.txt

COPY src/ .
#RUN python app/prestart.py

EXPOSE 3000
