# Gemma Embedding & Reranking Service

A high-performance Node.js service built with TypeScript that provides text embedding generation and document reranking capabilities using the Gemma 300M model from Hugging Face Transformers.

## Features

- 🚀 **Fast embedding generation** using the Gemma 300M ONNX model
- 📊 **Document reranking** based on semantic similarity (returns all indices)
- 🔒 **SSL/TLS support** for secure HTTPS connections
- 🐳 **Docker support** for easy deployment
- 💪 **TypeScript** for better type safety and developer experience
- 🛡️ **Production-ready** with security headers, compression, and error handling
- 📈 **Health checks** for monitoring
- 🔄 **Graceful shutdown** handling

## Quick Start with Docker

### Using Docker Compose (Recommended)

```bash
# Clone or create the project directory
git clone <repository-url> # or create the files manually
cd gemma-embed-service

# Start the service
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs -f
```

### Using Docker directly

```bash
# Build the image
docker build -t gemma-embed-service .

# Run the container
docker run -p 8082:8082 --name gemma-embed gemma-embed-service

# Run in background
docker run -d -p 8082:8082 --name gemma-embed gemma-embed-service
```

## Local Development

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Development with auto-reload
npm run dev

# Build TypeScript
npm run build

# Start production server
npm start
```

## API Documentation

### Base URL
```
http://localhost:8082
```

### Health Check

**GET** `/health`

Check service status and model initialization.

**Response:**
```json
{
  "status": "ok",
  "initialized": true,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### Generate Embeddings

**POST** `/embed`

Generate vector embeddings for text input.

**Request Body:**
```json
{
  "input": "Your text to embed"
}
```

or for multiple texts:
```json
{
  "input": ["First text", "Second text", "Third text"]
}
```

**Response:**
```json
{
  "embeddings": [
    [0.1, 0.2, 0.3, ...],  // 300-dimensional vectors
    [0.4, 0.5, 0.6, ...]
  ],
  "count": 2
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8082/embed \
  -H "Content-Type: application/json" \
  -d '{"input": "What is machine learning?"}'
```

### Rerank Documents

**POST** `/rerank`

Rerank documents based on their semantic similarity to a query.

**Request Body:**
```json
{
  "query": "Which planet is known as the Red Planet?",
  "documents": [
    "Venus is often called Earth's twin due to its similar size.",
    "Mars, known for its reddish appearance due to iron oxide on its surface.",
    "Jupiter is the largest planet in our solar system."
  ]
}
```

**Response:**
```json
{
  "ranking": [1, 2, 0]  // Indices in order of relevance (all indices returned)
}
```

The ranking array contains all document indices ordered by relevance (most relevant first).

**cURL Example:**
```bash
curl -X POST http://localhost:8082/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "space exploration",
    "documents": [
      "Cooking recipes for dinner",
      "NASA Mars rover discoveries",
      "Stock market trends"
    ]
  }'
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8082` | Server port |
| `NODE_ENV` | `development` | Environment mode |
| `USE_SSL` | `false` | Enable HTTPS/SSL support |
| `SSL_KEY_PATH` | `certs/key.pem` | Path to SSL private key file |
| `SSL_CERT_PATH` | `certs/cert.pem` | Path to SSL certificate file |

### Docker Environment

When using Docker, you can override environment variables:

```bash
docker run -p 8082:8082 -e PORT=9000 gemma-embed-service
```

Or in docker-compose.yml:
```yaml
environment:
  - PORT=9000
  - NODE_ENV=production
  - USE_SSL=true
```

### SSL/TLS Configuration

The service supports HTTPS/SSL for secure connections. To enable SSL:

#### 1. Generate SSL Certificates

For development, use the provided script to generate self-signed certificates:

```bash
# Make the script executable and run it
chmod +x generate-ssl-certs.sh
./generate-ssl-certs.sh
```

This creates:
- `certs/key.pem` - Private key
- `certs/cert.pem` - Certificate

#### 2. Enable SSL

Set the environment variable:

```bash
export USE_SSL=true
npm start
```

Or with Docker:

```bash
docker run -p 8082:8082 -e USE_SSL=true -v $(pwd)/certs:/app/certs gemma-embed-service
```

#### 3. Custom Certificate Paths

```bash
export USE_SSL=true
export SSL_KEY_PATH=/path/to/private.key
export SSL_CERT_PATH=/path/to/certificate.crt
```

**Note**: For production, use certificates from a trusted Certificate Authority instead of self-signed certificates.

## Performance & Scaling

### Model Loading
- The Gemma model (~300MB) loads on first request or server startup
- Subsequent requests are served from memory
- Initial model loading may take 30-60 seconds depending on hardware

### Resource Requirements
- **Memory**: 2-4GB RAM (model + overhead)
- **CPU**: 2+ cores recommended
- **Storage**: 1GB for model and dependencies

### Production Considerations
- Use a reverse proxy (nginx) for load balancing
- Consider horizontal scaling for high traffic
- Monitor memory usage and implement proper logging
- Set up persistent health checks

## Development

### Project Structure
```
gemma-embed-service/
├── src/
│   └── server.ts          # Main TypeScript server
├── dist/                  # Compiled JavaScript (generated)
├── package.json
├── tsconfig.json
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### Scripts

```bash
npm run build      # Compile TypeScript
npm run dev        # Development with hot reload
npm run start      # Start production server
npm run clean      # Clean build directory
npm run docker:build  # Build Docker image
npm run docker:run    # Run Docker container
```

### Adding Features

1. Extend the `EmbeddingService` class in `src/server.ts`
2. Add new endpoints following the existing pattern
3. Update TypeScript interfaces for type safety
4. Test locally with `npm run dev`
5. Rebuild Docker image for deployment

## Troubleshooting

### Common Issues

**Model Loading Fails**
- Ensure stable internet connection for initial download
- Check available disk space (>1GB required)
- Verify Node.js version (18+ required)

**High Memory Usage**
- Normal behavior: model requires 2-4GB RAM
- Consider increasing Docker memory limits
- Monitor with `docker stats` or system monitoring

**Slow Response Times**
- First request is slower due to model initialization
- Consider pre-warming the model on startup
- Check CPU resources and consider scaling

**Port Already in Use**
```bash
# Find process using port 8082
lsof -i :8082

# Kill process if needed
kill -9 <PID>

# Or use different port
PORT=8083 npm start
```

### Health Monitoring

```bash
# Check service health
curl http://localhost:8082/health

# Docker container health
docker inspect gemma-embed --format='{{.State.Health.Status}}'

# View container logs
docker logs gemma-embed
```

## API Rate Limiting

Currently, no rate limiting is implemented. For production use, consider adding:
- Rate limiting middleware
- Request queuing for high loads
- Authentication/API keys

## Security

The service includes basic security measures:
- Helmet.js for security headers
- CORS support
- Non-root Docker user
- Input validation
- Error handling without information leakage

For production, also consider:
- API authentication
- Request size limits (currently 10MB)
- Network security (VPC, firewalls)
- Regular security updates

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker logs
3. Open an issue in the repository

---

**Happy embedding! 🚀**
