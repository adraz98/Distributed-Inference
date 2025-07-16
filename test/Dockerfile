FROM python:3.10-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir requests

###############  production (tests usually run once) #################
FROM base AS prod
COPY test /app
CMD ["python","test_client.py"]

###############  development (optional) ##############################
FROM base AS dev
CMD ["python","test_client.py"]