# Use a reasonably recent version of Python (as of Sep 2024). The -bookworm
# variant ensures that this will run on both x86 and arm64 machines.
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# Install dependencies
RUN pip3 install flask
RUN pip3 install --upgrade google-cloud-vision
RUN pip3 install gunicorn
# Install Google Cloud SDK
RUN curl -sSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y google-cloud-sdk


