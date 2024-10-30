import os
import boto3
import yaml
import base64
from cryptography.fernet import Fernet

def get_aws_credentials():
    # Read from environment variable
    encrypted_creds = os.environ.get('AWS_CREDENTIALS')
    
    # Or read from mounted volume
    creds_path = '/var/run/secrets/aws/credentials'
    if os.path.exists(creds_path):
        with open(creds_path, 'r') as f:
            encrypted_creds = f.read()
    
    # Decrypt credentials using your encryption key
    # Note: Key should be securely managed, e.g., through AWS KMS
    encryption_key = os.environ.get('ENCRYPTION_KEY')
    f = Fernet(encryption_key)
    decrypted_creds = f.decrypt(encrypted_creds.encode())
    
    # Parse YAML credentials
    creds = yaml.safe_load(decrypted_creds)
    
    # Initialize AWS session
    session = boto3.Session(
        aws_access_key_id=creds['aws_access_key_id'],
        aws_secret_access_key=creds['aws_secret_access_key'],
        region_name=creds['region']
    )
    
    return session

# Usage example
def main():
    try:
        aws_session = get_aws_credentials()
        # Use the session to create AWS service clients
        s3_client = aws_session.client('s3')
        # Your application logic here
    except Exception as e:
        print(f"Error loading credentials: {e}")
        raise

if __name__ == "__main__":
    main()