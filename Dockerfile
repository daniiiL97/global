FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 3478
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=3478", "--server.address=0.0.0.0"]
