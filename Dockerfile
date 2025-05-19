FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY api_requirements.txt .
RUN pip install --no-cache-dir -r api_requirements.txt

# Copy application code
COPY app.py .
COPY templates/ templates/
COPY deployment/ deployment/
COPY model.py .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 5000

# Environment variables
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"] 