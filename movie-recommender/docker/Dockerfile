# Container for the CineMatch Streamlit demo
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the model artifacts at image-build time using the bundled sample data
# (swap data/raw/*.csv for the real TMDB CSVs before building for production use)
RUN python scripts/make_sample_dataset.py && \
    python main.py --config config/config.yaml

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
