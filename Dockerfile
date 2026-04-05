FROM python:3.10-slim

# 1. Create a non-root user (Hugging Face requires user ID 1000)
RUN useradd -m -u 1000 user
USER user

# 2. Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=8000

# 3. Set standard working directory
WORKDIR $HOME/app

# 4. Copy and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 5. Copy your application code
COPY --chown=user . .

# 6. Expose the port
EXPOSE $PORT

# 7. Start the FastAPI server using the PORT variable
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
