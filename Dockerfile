# Use an official Tensorflow runtime as a parent image
FROM tensorflow/tensorflow:latest

# Set the working directory to /app
WORKDIR /nanogpt

COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pyyaml

# RUN chmod +x scripts/master.sh
# RUN chmod +x scripts/worker.sh

# Run app.py when the container launches
CMD ["python", "train.py"]
