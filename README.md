# Gemma Embedding & Reranking Service

A high-performance Node.js service built with TypeScript that provides text embedding generation and document reranking capabilities using the Gemma 300M model from Hugging Face Transformers.

## Features

- 🚀 **Fast embedding generation** using the Gemma 300M ONNX model
- 🔗 **OpenAI API compatibility** - works with existing OpenAI SDK and libraries
- 📏 **Matryoshka Representation Learning (MRL)** - native support for 768, 512, 256, 128 dimensions
- 📊 **Document reranking** based on semantic similarity (returns all indices)
- 🔒 **SSL/TLS support** for secure HTTPS connections
- 🔐 **API Key authentication** for secure access control
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

# IMPORTANT: Set your API key in docker-compose.yml
# Edit the API_KEY environment variable before starting

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

# Run the container (replace YOUR_API_KEY with your actual key)
docker run -p 8082:8082 -e API_KEY="YOUR_API_KEY" --name gemma-embed gemma-embed-service

# Run in background
docker run -d -p 8082:8082 -e API_KEY="YOUR_API_KEY" --name gemma-embed gemma-embed-service
```

## Local Development

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Set your API key
export API_KEY="your-secret-api-key-here"

# Development with auto-reload
npm run dev

# Build TypeScript
npm run build

# Start production server
npm start
```

## API Documentation

This service follows the OpenAI Embeddings API standard format for compatibility with existing tools and libraries.

### Base URL
```
http://localhost:8082
```

### Health Check

**GET** `/health`

Check service status and model initialization. **No authentication required.**

**Response:**
```json
{
  "status": "ok",
  "initialized": true,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### List Models (OpenAI Compatible)

**GET** `/models` or **GET** `/v1/models`

List available embedding models in OpenAI-compatible format. **No authentication required.**

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "embeddinggemma-300m",
      "object": "model",
      "created": 1704067200,
      "owned_by": "onnx-community"
    }
  ]
}
```

### Generate Embeddings (OpenAI Compatible)

**POST** `/v1/embeddings`

Generate vector embeddings for text input using OpenAI-compatible format. **Requires API key authentication.**

**Headers:**
```
Authorization: Bearer YOUR_API_KEY
# OR
Authorization: YOUR_API_KEY
Content-Type: application/json
```

**Request Body:**
```json
{
  "input": "Your text to embed",
  "model": "embeddinggemma-300m",
  "encoding_format": "float"
}
```

**With multiple texts and MRL dimension reduction:**
```json
{
  "input": ["First text", "Second text", "Third text"],
  "model": "embeddinggemma-300m",
  "dimensions": 256,
  "encoding_format": "float"
}
```

**Request Parameters:**
- `input` (required): String or array of strings to embed
- `model` (required): Model identifier (`embeddinggemma-300m`)
- `dimensions` (optional): Number of dimensions for output embeddings (1-768, default: 768)
  - **Recommended MRL dimensions**: 768 (full), 512, 256, 128 for optimal quality
- `encoding_format` (optional): Format for embeddings (`float` only, default: `float`)
- `user` (optional): User identifier for tracking

**Response (OpenAI Compatible):**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1, 0.2, 0.3, ...]
    },
    {
      "object": "embedding", 
      "index": 1,
      "embedding": [0.4, 0.5, 0.6, ...]
    }
  ],
  "model": "embeddinggemma-300m",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

**cURL Examples:**
```bash
# Basic embedding request
curl -X POST http://localhost:8082/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key" \
  -d '{
    "input": "What is machine learning?",
    "model": "embeddinggemma-300m"
  }'

# With MRL dimension reduction
curl -X POST http://localhost:8082/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: your-secret-api-key" \
  -d '{
    "input": "What is machine learning?",
    "model": "embeddinggemma-300m",
    "dimensions": 256
  }'

# Multiple texts
curl -X POST http://localhost:8082/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key" \
  -d '{
    "input": ["First text", "Second text"],
    "model": "embeddinggemma-300m"
  }'
```

### Generate Embeddings (Legacy Format)

**POST** `/embed`

*Legacy endpoint for backward compatibility. Use `/v1/embeddings` for new integrations.*

**Request Body:**
```json
{
  "input": "Your text to embed"
}
```

**Response:**
```json
{
  "embeddings": [
    [0.1, 0.2, 0.3, ...]  // 768-dimensional vectors (default)
  ],
  "count": 1
}
```

### Rerank Documents

**POST** `/rerank`

Rerank documents based on their semantic similarity to a query. **Requires API key authentication.**

**Headers:**
```
Authorization: Bearer YOUR_API_KEY
# OR
Authorization: YOUR_API_KEY
Content-Type: application/json
```

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
  -H "Authorization: Bearer your-secret-api-key" \
  -d '{
    "query": "space exploration",
    "documents": [
      "Cooking recipes for dinner",
      "NASA Mars rover discoveries",
      "Stock market trends"
    ]
  }'
```

## OpenAI SDK Compatibility

This service is compatible with the OpenAI SDK and other libraries that follow the OpenAI API format:

### JavaScript/TypeScript (OpenAI SDK)
```javascript
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: "your-secret-api-key",
  baseURL: "http://localhost:8082/v1", // Point to your local service
});

const embedding = await openai.embeddings.create({
  model: "embeddinggemma-300m",
  input: "Your text string goes here",
  encoding_format: "float",
});

// With MRL dimension reduction (optimal quality)
const shortEmbedding = await openai.embeddings.create({
  model: "embeddinggemma-300m", 
  input: "Your text string goes here",
  dimensions: 256, // MRL-trained dimension
});
```

### Python (OpenAI Library)
```python
from openai import OpenAI

client = OpenAI(
    api_key="your-secret-api-key",
    base_url="http://localhost:8082/v1"
)

response = client.embeddings.create(
    input="Your text string goes here",
    model="embeddinggemma-300m"
)

print(response.data[0].embedding)

# With MRL dimension reduction (optimal quality)
response = client.embeddings.create(
    input="Your text string goes here",
    model="embeddinggemma-300m",
    dimensions=512  # MRL-trained dimension
)
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
| `API_KEY` | **(required)** | API key for authentication |

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
  - API_KEY=your-secret-api-key
```

### Authentication

The service requires an API key for accessing the `/v1/embeddings`, `/embed`, and `/rerank` endpoints. The `/health` and `/models` endpoints remain public for monitoring purposes.

#### Setting up API Key

**Environment Variable:**
```bash
export API_KEY="your-secret-api-key-here"
npm start
```

**Docker:**
```bash
docker run -p 8082:8082 -e API_KEY="your-secret-api-key-here" gemma-embed-service
```

**Docker Compose:**
Update your `docker-compose.yml`:
```yaml
environment:
  - API_KEY=your-secret-api-key-here
```

#### Using the API Key

Include the API key in the `Authorization` header:

**Option 1: Bearer Token Format (Recommended)**
```bash
curl -H "Authorization: Bearer your-secret-api-key-here" ...
```

**Option 2: Direct Key Format**
```bash
curl -H "Authorization: your-secret-api-key-here" ...
```

Both formats are supported and equivalent.

#### Error Responses

**Missing API Key (401 Unauthorized):**
```json
{
  "error": "Unauthorized",
  "message": "Authorization header is required"
}
```

**Invalid API Key (401 Unauthorized):**
```json
{
  "error": "Unauthorized",
  "message": "Invalid API key"
}
```

**Server Configuration Error (500):**
```json
{
  "error": "Server configuration error",
  "message": "API_KEY environment variable not set"
}
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

## Model Information

### Available Models
- **embeddinggemma-300m**: EmbeddingGemma model trained with Matryoshka Representation Learning
  - **Full Dimensions**: 768 (default)
  - **MRL Dimensions**: 768, 512, 256, 128 (trained with MRL for optimal quality)
  - **Max Input Tokens**: ~8000 (estimated)
  - **Context Length**: Variable (depends on tokenizer)
  - **Use Case**: General-purpose text embeddings with flexible dimensionality

### Matryoshka Representation Learning (MRL) Support
This model has been specifically trained with Matryoshka Representation Learning:
- **Trained Dimensions**: 768 (full), 512, 256, 128 - these provide optimal quality
- **Custom Dimensions**: Any dimension from 1-768 supported via proper MRL method
- **Implementation**: Uses layer normalization → slicing → L2 normalization (not simple truncation)
- **Quality**: All dimensions maintain high quality through proper MRL processing
- **Use Cases**: Trade-off between embedding quality, processing speed, and storage requirements
- **Recommendation**: Use 768, 512, 256, or 128 for best results, but any dimension works well

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

# List available models
curl http://localhost:8082/models

# Docker container health
docker inspect gemma-embed --format='{{.State.Health.Status}}'

# View container logs
docker logs gemma-embed
```

## API Rate Limiting

Currently, no rate limiting is implemented. For production use, consider adding:
- Rate limiting middleware
- Request queuing for high loads
- Per-API-key rate limiting

## Security

The service includes comprehensive security measures:
- **API Key Authentication** for endpoint access control
- **Helmet.js** for security headers
- **CORS support** for cross-origin requests
- **Non-root Docker user** for container security
- **Input validation** and sanitization
- **Error handling** without information leakage
- **HTTPS/SSL support** for encrypted connections

For production, also consider:
- **Strong API keys** (use cryptographically secure random strings)
- **API key rotation** policies
- **Request size limits** (currently 10MB)
- **Network security** (VPC, firewalls, IP whitelisting)
- **Regular security updates** and monitoring
- **Rate limiting** per API key

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker logs
3. Open an issue in the repository

---

**Happy embedding! 🚀**
