FROM python:3.8-slim-buster

WORKDIR /usr/src/app

COPY /services/backend/requirements.txt ./
ENV  PYTHONPATH=/usr/app/app  


RUN pip3 install --no-cache-dir --upgrade pip \
&& pip3 install --no-cache-dir -r requirements.txt

COPY ../.. .
#RUN python app/prestart.py

EXPOSE 3000

