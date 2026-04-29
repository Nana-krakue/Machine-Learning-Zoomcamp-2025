FROM python:3.12-slim

RUN pip install pipenv

WORKDIR /app
# COPY ["Pipfile", "Pipfile.lock", "./"]
COPY ["requirements.txt", "./"]

# RUN pipenv install --system --deploy
# RUN pip install flask gunicorn scikit-learn==0.24.2 numpy==1.26.4

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# COMMAND TO RUN:
#   docker build -t zoomcamp-test .   
#   docker run -it --rm -p 9696:9696 zoomcamp-test

RUN mkdir ./model
COPY ["predict.py", "./"]
COPY ["./model/model_C=1.0.bin", "./model/"]

# COMMAND TO RUN:
#    gunicorn --bind=0.0.0.0:9696 predict:app

EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]