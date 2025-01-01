1. Set Up AWS Secret Manager Secret
Make sure the Snowflake credentials are stored in AWS Secrets Manager in JSON format, e.g.:

json
Copy code
{
  "username": "your_snowflake_user",
  "password": "your_secure_password",
  "account": "your_snowflake_account",
  "database": "your_database",
  "warehouse": "your_warehouse",
  "role": "your_role"
}
Ensure your application or pods can access this secret.

2. Deploy External Secrets Operator
You need the External Secrets Operator (ESO) to manage secrets in Kubernetes using AWS Secrets Manager.

Install External Secrets Operator:
Run the following commands in your Kubernetes cluster:

bash
Copy code
kubectl apply -f https://github.com/external-secrets/external-secrets/releases/latest/download/install.yaml
This will install the Custom Resource Definitions (CRDs) and necessary resources.

3. Set Up IAM Role for Service Accounts (IRSA)
To grant Kubernetes pods running on EKS access to AWS Secrets Manager securely, you must use IAM Roles for Service Accounts.

Create an IAM role with a policy to access the AWS Secrets Manager:

Example IAM Policy (secrets-access-policy.json):

json
Copy code
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:region:account-id:secret:secret-name"
    }
  ]
}
Attach the policy to a role and link it to your Kubernetes service account (IRSA).

Use eksctl to set it up:

bash
Copy code
eksctl create iamserviceaccount \
  --name external-secrets-sa \
  --namespace default \
  --cluster your-cluster-name \
  --attach-policy-arn arn:aws:iam::account-id:policy/secrets-access-policy \
  --approve
4. Create Kubernetes External Secret
Once External Secrets is installed, create a custom resource called ExternalSecret. This resource will fetch credentials from AWS Secrets Manager and create a Kubernetes Secret.

Example externalsecret.yaml:

yaml
Copy code
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: snowflake-secret
  namespace: default
spec:
  refreshInterval: "1h"
  secretStoreRef:
    name: aws-secret-store
    kind: SecretStore
  target:
    name: snowflake-secret
    creationPolicy: Owner
  data:
  - secretKey: username
    remoteRef:
      key: snowflake/aws/secret  # Replace with the AWS Secrets Manager secret name
      property: username
  - secretKey: password
    remoteRef:
      key: snowflake/aws/secret
      property: password
  - secretKey: account
    remoteRef:
      key: snowflake/aws/secret
      property: account
5. Configure SecretStore for AWS Secrets Manager
The SecretStore connects External Secrets to AWS Secrets Manager.

Create secretstore.yaml:


apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secret-store
  namespace: default
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2  # Replace with your AWS region
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa  # Service account created for IRSA
6. Deploy and Verify
Apply the SecretStore and ExternalSecret:

kubectl apply -f secretstore.yaml
kubectl apply -f externalsecret.yaml
Verify the Secret was created in Kubernetes:

bash
Copy code
kubectl get secret snowflake-secret
7. Use the Kubernetes Secret in EKS Pods
Mount the Kubernetes Secret (created via External Secrets) in your EKS pod or workload as environment variables or as a file.

Example Deployment:
yaml
Copy code
apiVersion: apps/v1
kind: Deployment
metadata:
  name: snowflake-client
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: snowflake-client
  template:
    metadata:
      labels:
        app: snowflake-client
    spec:
      serviceAccountName: external-secrets-sa  # IRSA service account
      containers:
      - name: snowflake-app
        image: your-snowflake-client-image
        env:
        - name: SNOWFLAKE_USER
          valueFrom:
            secretKeyRef:
              name: snowflake-secret
              key: username
        - name: SNOWFLAKE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: snowflake-secret
              key: password
        - name: SNOWFLAKE_ACCOUNT
          valueFrom:
            secretKeyRef:
              name: snowflake-secret
              key: account
8. Test the Snowflake Connection
Once the pod is running:

Access the pod shell:

bash
Copy code
kubectl exec -it <pod-name> -- bash
Use the Snowflake client or library (Python snowflake-connector-python) to test the connection.

For example, in Python:

python
Copy code
import snowflake.connector
import os

conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT")
)
print("Connection Successful!")
conn.close()
Summary
    AWS Secrets Manager securely stores Snowflake credentials.
    IAM Role for Service Accounts (IRSA) allows secure access to AWS Secrets Manager.
External Secrets Operator fetches credentials and creates Kubernetes secrets.
The Snowflake credentials are mounted in pods securely as environment variables.
This approach ensures credentials are managed securely and dynamically without hardcoding sensitive information in Kubernetes manifests.