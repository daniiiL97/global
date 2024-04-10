FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501"]
EXPOSE 8501
