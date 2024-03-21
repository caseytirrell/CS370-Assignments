# Start with a PyTorch base image that is compatible with your system
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /workspace

# Install GDAL dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin

# Set the GDAL environment variable
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy the requirements file into the container
COPY requirements.txt ./

# Install Python dependencies from requirements.txt
# Make sure you don't include gdal in the requirements.txt as it's already installed
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run app.py when the container launches
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--
