FROM python:3.10

WORKDIR /app

ADD data /app/data
ADD models /app/models
ADD server /app/server

ADD requirements.txt /requirements.txt

RUN pip3 install -r /requirements.txt

CMD [ "python", "/app/server/regression.py" ]
