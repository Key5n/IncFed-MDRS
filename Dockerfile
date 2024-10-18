FROM python:3.13.0-bookworm

WORKDIR /workspace

RUN apt update && \
    apt upgrade -y && \
    cd workspace && \
    pip install .

CMD [ "bash" ]
