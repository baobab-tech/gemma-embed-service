#!/bin/bash

# Script to generate self-signed SSL certificates for development

echo "Generating self-signed SSL certificates for development..."

# Create certs directory if it doesn't exist
mkdir -p certs

# Generate private key
openssl genrsa -out certs/key.pem 2048

# Generate certificate signing request
openssl req -new -key certs/key.pem -out certs/csr.pem -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Generate self-signed certificate
openssl x509 -req -in certs/csr.pem -signkey certs/key.pem -out certs/cert.pem -days 365

# Clean up CSR file
rm certs/csr.pem

# Set appropriate permissions for container access
chmod 644 certs/key.pem certs/cert.pem

echo "SSL certificates generated successfully!"
echo "  Private key: certs/key.pem"
echo "  Certificate: certs/cert.pem"
echo "  Permissions set to 644 for container compatibility"
echo ""
echo "To use SSL, set the environment variable: USE_SSL=true"
echo ""
echo "Note: These are self-signed certificates for development only."
echo "For production, use certificates from a trusted Certificate Authority."

chmod +x generate-ssl-certs.sh
