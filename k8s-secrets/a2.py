import base64
from cryptography.fernet import Fernet
import yaml

# 1. Start with original YAML credentials
original_creds = """
aws_access_key_id: AKIAXXXXXXXXXXXXXXXX
aws_secret_access_key: abc123xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
region: us-west-2
"""

# 2. Generate encryption key (in practice, store this securely)
encryption_key = Fernet.generate_key()
f = Fernet(encryption_key)

# 3. First encrypt the credentials
encrypted_data = f.encrypt(original_creds.encode())

# 4. Then base64 encode the encrypted data for Kubernetes
k8s_safe_data = base64.b64encode(encrypted_data).decode()

# This is what goes in your Kubernetes secret YAML
print("Value for Kubernetes secret:")
print(f"credentials: {k8s_safe_data}")

# --- Later, when decrypting in your application ---

# 1. Get base64 encoded and encrypted data from Kubernetes
encrypted_from_k8s = k8s_safe_data

# 2. Base64 decode (Kubernetes does this automatically when mounting)
encrypted_bytes = base64.b64decode(encrypted_from_k8s)

# 3. Decrypt using the same key
decrypted_data = f.decrypt(encrypted_bytes)

# 4. Parse the YAML
credentials = yaml.safe_load(decrypted_data)

print("\nRecovered credentials:")
print(credentials)