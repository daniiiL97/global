FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=3478", "--server.address=0.0.0.0"]
EXPOSE 3478
