FROM python:3.9-slim-buster

LABEL version=0.1

ENV PYTHONUNBUFFERED True

RUN apt-get update && \
    apt-get -y install g++ && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/libc/gcc/aarch64-linux-gnu/8/cc1plus /usr/local/bin/

RUN mkdir app
ENV APP_HOME /app
WORKDIR $APP_HOME

ENV PORT 5587

ENV PATH="${PATH}:/root/.local/bin"
ENV PYTHONPATH=.

COPY services/backend/requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r ./requirements.txt

COPY services/backend/src/ .
COPY services/utils/ utils/

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
