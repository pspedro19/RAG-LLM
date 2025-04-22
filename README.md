##########

# Microservices Architecture for Language Model Project

## Overview

This project utilizes a microservices architecture to implement and manage a language model. The architecture is designed to facilitate inter-service communication, track machine learning experiments, manage model lifecycles, store artifacts, and monitor tasks. Below is a detailed explanation of each component and how they interact.

$$$$$$$$$$$$$$$$$


# Microservices Architecture for Language Model Project

## Overview

This project utilizes a microservices architecture to implement and manage a language model. The architecture is designed to facilitate inter-service communication, track machine learning experiments, manage model lifecycles, store artifacts, and monitor tasks. Below is a detailed explanation of each component and how they interact.

$$$$$$$$$$$$$$$$$

Using apt (Recommended)
Update the package database:

```bash
sudo apt update -y
```
Install Docker Compose:

```bash
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
```
Using snap
Install Docker using snap:

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

```bash
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
```

```bash
sudo apt update -y
```

```bash
sudo apt install -y docker-ce
```
```bash
sudo systemctl start docker
```

```bash
sudo systemctl enable docker
```

```bash
sudo systemctl status docker
```

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
```


```bash
sudo chmod +x /usr/local/bin/docker-compose
```

```bash
docker-compose --version
```

```bash
sudo usermod -aG docker $USER
```

```bash
docker-compose build
```


Download the Docker Compose binary:

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
```
Apply executable permissions to the binary:

```bash
sudo chmod +x /usr/local/bin/docker-compose
```
Create a symbolic link to make Docker Compose available in the system's PATH:

```bash
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
```
Verify that Docker Compose is installed correctly:

```bash
docker-compose --version
```
After installing Docker Compose using either method, you should be able to run `docker-compose build`.



## MLflow CONFIG

If MLflow is failing to connect because the database `mlflow_db` does not exist, you need to create this database in your PostgreSQL instance.

### Create the `mlflow_db` Database:

1. Connect to your PostgreSQL server using a client like `psql` or any GUI tool.
    ```sql
    docker exec -it postgres_airflow psql -U airflow;
    ```

3. Create the database `mlflow_db`:
    ```sql
    CREATE DATABASE mlflow_db;
    ```
4. Ensure Correct Database Configuration: Verify that the MLflow configuration in your Docker Compose file or environment variables points to the correct database:
    ```yaml
    environment:
      - MLFLOW_TRACKING_URI=postgresql://username:password@postgres/mlflow_db
    ```
    ```sh
    docker restart mlflow_tracking
    ```
    


## Full Steps

1. **Rebuild the Docker Images:**
    ```sh
    docker-compose build
    ```
2. **Create the `mlflow_db` Database:**
    - Connect to PostgreSQL and create the database:
    ```sh
    docker-compose exec postgres_airflow psql -U airflow
    CREATE DATABASE mlflow_db;
    ```
3. **Restart Services:**
    - Restart your Docker Compose services to apply the changes:
    ```sh
    docker-compose up -d
    ```

4. In the root folder of this repository, execute:
    ```sh
    docker compose --profile all up
    ```

Once all the services are up and running (verify with the command docker ps -a that all services are healthy or check in Docker Desktop), you can access the various services through:

   - Apache Airflow: http://localhost:8080
   - MLflow: http://localhost:5000
   - MinIO (service): http://localhost:9000 
   - MinIO (Buckets Administration): http://localhost:9001
   - Django Chat Interface: http://localhost:8800
   - FastAPI: http://localhost:8000

API Docs: http://localhost:8800/docs
If you are using a server external to your work computer, replace localhost with its IP address (it can be a private IP if your server is on your LAN or a public IP otherwise; check firewalls or other rules that may prevent connections).

All ports and other configurations can be modified in the .env file. You are encouraged to experiment and learn by trial and error; you can always re-clone this repository.
## Architecture Components

1. **Chat**
   - Purpose: Manages the user interface and processes user requests.
   - Connection: Sends user requests to the REST API for processing.
2. **REST API (FastAPI)**
   - Purpose: Provides inter-service communication between the components of the architecture.
   - Connection: Receives requests from the Chat service and forwards them to MLflow for model interactions and artifact handling.
3. **MLflow**
   - Purpose: Tracks machine learning experiments and manages the model lifecycle.
   - Connection: Receives data from REST API, interacts with Apache Airflow to orchestrate workflows, stores and retrieves artifacts from MinIO, and utilizes PostgreSQL for metadata storage.
4. **Apache Airflow**
   - Purpose: Orchestrates workflows and schedules tasks.
   - Connection: Schedules tasks through a dedicated Scheduler, provides monitoring through a Webserver, sends and receives performance metrics to and from Wandb.
5. **Scheduler & Webserver**
   - Purpose: These components are part of Apache Airflow.
   - Scheduler: Manages the timing of workflows.
   - Webserver: Provides a GUI for monitoring the scheduled tasks.
6. **Wandb**
   - Purpose: Tracks the visualization and performance of models.
   - Connection: Receives data from Apache Airflow to visualize task performance.
7. **MinIO**
   - Purpose: S3-compatible storage used for managing ML artifacts.
   - Connection: Stores artifacts generated or used by MLflow.
8. **PostgreSQL**
   - Purpose: Database used for storing metadata and operational data.
   - Connection: Stores experiment and model data for MLflow, maintains operational data for Apache Airflow.

## Diagram Visualization

The microservices architecture diagram visually represents the interconnections between different services. Each service is depicted with its respective icon and connected through lines indicating data flow and interactions.

## Suggestions for Improvement

- **Security:** Implement security measures like API gateways or OAuth to manage access to the services.
- **Scalability:** Consider container orchestration solutions like Kubernetes for better scalability and management of services.
- **Monitoring:** Integrate comprehensive monitoring tools like Prometheus and Grafana for better insight into service performance and health.
- **Bidirectional Data Flow:** Review and ensure that bidirectional data flows are necessary and optimized for performance.
