FROM python:3.9
ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN mkdir /app
WORKDIR /app

RUN apt-get update

# Install pip and poetry
RUN pip install --upgrade pip
RUN pip install "poetry==1.1.8"

# Create layer for dependencies
COPY ./poetry.lock ./pyproject.toml ./

# Install python dependencies using poetry
RUN poetry config virtualenvs.create false
RUN poetry install

# Download the sentence transformer model
RUN mkdir /models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/msmarco-distilbert-dot-v5', cache_folder='/models')"

# Copy files to image
COPY ./src ./src
COPY ./cli ./cli
COPY ./data ./data

# Add Environment Variables to Run Locally 
ARG AWS_ACCESS_KEY_ID=
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID

ARG AWS_SECRET_ACCESS_KEY=
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# Run the indexer on the input s3 directory
CMD [ "sh", "./cli/run.sh" ]