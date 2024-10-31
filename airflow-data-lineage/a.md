This YAML configuration sets up a Kubernetes pipeline for data lineage using Airflow and OpenLineage. Here's a breakdown of the key components:
A dedicated namespace airflow-lineage is created to isolate the resources.
An Airflow webserver deployment is configured with the KubernetesExecutor and OpenLineage backend1. The OpenLineage URL is set to point to the Marquez service.
An Airflow scheduler deployment is set up similarly to the webserver.
A Marquez deployment is included to serve as the OpenLineage backend for storing and analyzing lineage data2
.
Services are created for the Airflow webserver and Marquez to enable network communication.