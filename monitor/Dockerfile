###############  common base ####################################
FROM python:3.10-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir streamlit requests pandas
EXPOSE 8501

# OS‑level tools needed
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

###############  production ######################################
FROM base AS prod
COPY monitor /app
CMD ["streamlit","run","dashboard.py","--server.port=8501","--server.address=0.0.0.0"]

###############  development (auto‑reload) #######################
FROM base AS dev
# live‑code mount, Streamlit reloads automatically when files change
#ENV STREAMLIT_SERVER_RUN_ON_SAVE=true
CMD ["streamlit","run","dashboard.py","--server.port=8501","--server.address=0.0.0.0"]
