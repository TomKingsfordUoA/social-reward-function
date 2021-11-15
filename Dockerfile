FROM python:3.7-buster
WORKDIR /app
COPY . /app
RUN apt update
RUN ./install_system_dependencies.sh
RUN apt install -y sudo portaudio19-dev python-pyaudio python3-pyaudio
RUN python -m venv venv
RUN venv/bin/python -m pip install .
ENTRYPOINT ["./entrypoint.sh"]
