# Use an official Python runtime as a parent image
FROM python:3

# Set the working directory to /app
WORKDIR /usr/src/pipe

# Copy the current directory contents into the container at /app
ADD . /usr/src/pipe

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "run/debug_pipeline.sh"]