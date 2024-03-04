FROM postgres:latest

# Install the packages needed to build pgvector
RUN apt-get update && apt-get install -y build-essential postgresql-server-dev-all git curl

# Clone and install pgvector
RUN git clone https://github.com/ankane/pgvector.git && \
    cd pgvector && \
    make && \
    make install

# Enable the pgvector extension
RUN echo "shared_preload_libraries = 'pgvector'" >> /usr/share/postgresql/postgresql.conf.sample

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
