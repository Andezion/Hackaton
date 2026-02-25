FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generate.py analyze.py report.py providers.py ./

# Create output directories so volume mounts work cleanly
RUN mkdir -p data results

CMD ["python", "generate.py"]
