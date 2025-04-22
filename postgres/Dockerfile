FROM postgres:13

# Actualizar e instalar dependencias necesarias para compilar pgvector
RUN apt-get update && \
    apt-get install -y \
      postgresql-server-dev-13 \
      make \
      gcc \
      git && \
    # Clonar, compilar e instalar pgvector
    git clone https://github.com/pgvector/pgvector.git /tmp/pgvector && \
    cd /tmp/pgvector && \
    make && \
    make install && \
    # Limpiar
    cd / && rm -rf /tmp/pgvector && \
    apt-get remove -y postgresql-server-dev-13 make gcc git && \
    apt-get autoremove -y && \
    apt-get clean
