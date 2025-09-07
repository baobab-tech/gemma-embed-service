import express, { type Request, type Response, type NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import https from 'https';
import fs from 'fs';
import path from 'path';
import { AutoModel, AutoTokenizer, matmul, layer_norm } from '@huggingface/transformers';

// OpenAI-compatible interfaces
interface EmbeddingsRequest {
  input: string | string[];
  model: string;
  encoding_format?: 'float' | 'base64';
  dimensions?: number;
  user?: string;
}

interface EmbeddingData {
  object: 'embedding';
  index: number;
  embedding: number[];
}

interface EmbeddingsResponse {
  object: 'list';
  data: EmbeddingData[];
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

interface ModelInfo {
  id: string;
  object: 'model';
  created: number;
  owned_by: string;
}

interface ModelsResponse {
  object: 'list';
  data: ModelInfo[];
}

interface RerankRequest {
  query: string;
  documents: string[];
}

interface RankingItem {
  index: number;
  score: number;
}

class EmbeddingService {
  private model: unknown = null;
  private tokenizer: unknown = null;
  private isInitialized: boolean = false;
  private initPromise: Promise<void> | null = null;
  private readonly prefixes = {
    query: "task: search result | query: ",
    document: "title: none | text: ",
  };
  private readonly MODEL_INFO: ModelInfo = {
    id: 'embeddinggemma-300m',
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'onnx-community'
  };
  private readonly MAX_DIMENSIONS = 768; // EmbeddingGemma full dimensions
  private readonly SUPPORTED_DIMENSIONS = [768, 512, 256, 128]; // MRL-trained dimensions

  async initialize(): Promise<void> {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = this._doInitialize();
    return this.initPromise;
  }

  private async _doInitialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log('Initializing embedding model...');
    console.time('Model initialization');
    const model_id = "onnx-community/embeddinggemma-300m-ONNX";
    
    try {
      this.tokenizer = await AutoTokenizer.from_pretrained(model_id);
      this.model = await AutoModel.from_pretrained(model_id, {
        dtype: "fp32",
      });
      
      this.isInitialized = true;
      console.timeEnd('Model initialization');
      console.log('Model initialized successfully');
    } catch (error) {
      console.timeEnd('Model initialization');
      console.error('Failed to initialize model:', error);
      throw error;
    }
  }

  async getEmbeddings(texts: string[], dimensions?: number): Promise<number[][]> {
    await this.initialize();
    
    const startTime = Date.now();
    console.time(`Embedding generation (${texts.length} texts)`);
    
    // Process all texts at once for maximum performance
    const processedTexts = texts.map(text => this.prefixes.document + text);
    
    const inputs = await (this.tokenizer as any)(processedTexts, { padding: true });
    const { sentence_embedding } = await (this.model as any)(inputs);
    
    let embeddings = sentence_embedding;
    
    // Apply Matryoshka dimension reduction if requested
    if (dimensions && dimensions < this.MAX_DIMENSIONS) {
      // Proper MRL implementation following the standard approach:
      // 1. Layer normalization across embedding dimension
      // 2. Slice to desired matryoshka dimension
      // 3. L2 normalization for unit vectors
      embeddings = layer_norm(embeddings, [embeddings.dims[1]])
        .slice(null, [0, dimensions])
        .normalize(2, -1);
    }
    
    const allEmbeddings = embeddings.tolist();
    
    const duration = Date.now() - startTime;
    console.timeEnd(`Embedding generation (${texts.length} texts)`);
    console.log(`Generated embeddings for ${texts.length} texts in ${duration}ms (${(duration/texts.length).toFixed(1)}ms per text)`);
    
    return allEmbeddings;
  }

  getModelInfo(): ModelInfo {
    return { ...this.MODEL_INFO };
  }

  getMaxDimensions(): number {
    return this.MAX_DIMENSIONS;
  }

  getSupportedDimensions(): number[] {
    return [...this.SUPPORTED_DIMENSIONS];
  }

  // Simple token counting (rough estimate)
  private countTokens(text: string): number {
    // Rough estimation: ~4 characters per token on average
    return Math.ceil(text.length / 4);
  }

  getTotalTokens(texts: string[]): number {
    return texts.reduce((total, text) => total + this.countTokens(text), 0);
  }

  async rerank(query: string, documents: string[]): Promise<number[]> {
    await this.initialize();
    
    const startTime = Date.now();
    console.time(`Reranking (${documents.length} documents)`);
    
    // Process query embedding first
    const queryText = this.prefixes.query + query;
    const queryInputs = await (this.tokenizer as any)([queryText], { padding: true });
    const { sentence_embedding: queryEmbedding } = await (this.model as any)(queryInputs);
    const queryVector = queryEmbedding.tolist()[0];
    
    // Process all documents at once for maximum performance
    const processedDocuments = documents.map(doc => this.prefixes.document + doc);
    const inputs = await (this.tokenizer as any)(processedDocuments, { padding: true });
    const { sentence_embedding } = await (this.model as any)(inputs);
    const documentVectors = sentence_embedding.tolist();
    
    // Compute similarities for all documents
    const similarities: number[] = [];
    for (const docVector of documentVectors) {
      // Compute cosine similarity manually
      let dotProduct = 0;
      let queryMagnitude = 0;
      let docMagnitude = 0;
      
      for (let j = 0; j < queryVector.length; j++) {
        dotProduct += queryVector[j] * docVector[j];
        queryMagnitude += queryVector[j] * queryVector[j];
        docMagnitude += docVector[j] * docVector[j];
      }
      
      const similarity = dotProduct / (Math.sqrt(queryMagnitude) * Math.sqrt(docMagnitude));
      similarities.push(similarity);
    }
    
    // Create ranking with original indices
    const ranking: RankingItem[] = similarities
      .map((score, index) => ({ index, score }))
      .sort((a, b) => b.score - a.score);
    
    const duration = Date.now() - startTime;
    console.timeEnd(`Reranking (${documents.length} documents)`);
    console.log(`Reranked ${documents.length} documents in ${duration}ms (${(duration/documents.length).toFixed(1)}ms per document)`);
    
    return ranking.map(item => item.index);
  }

  get initialized(): boolean {
    return this.isInitialized;
  }
}

// Initialize service
const embeddingService = new EmbeddingService();

// Create Express app
const app: express.Application = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 8082;

// SSL Configuration
const SSL_KEY_PATH = process.env.SSL_KEY_PATH || path.join(__dirname, '../certs/key.pem');
const SSL_CERT_PATH = process.env.SSL_CERT_PATH || path.join(__dirname, '../certs/cert.pem');
const USE_SSL = process.env.USE_SSL === 'false' ? false : true; // Default to true, only false if explicitly set

// API Key Configuration
const API_KEY = process.env.API_KEY;

// Authentication middleware
const authenticateApiKey = (req: Request, res: Response, next: NextFunction) => {
  if (!API_KEY) {
    return res.status(500).json({ 
      error: 'Server configuration error',
      message: 'API_KEY environment variable not set'
    });
  }

  const authHeader = req.headers.authorization;
  if (!authHeader) {
    return res.status(401).json({ 
      error: 'Unauthorized',
      message: 'Authorization header is required'
    });
  }

  // Support both "Bearer <key>" and just "<key>" formats
  const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : authHeader;
  
  if (token !== API_KEY) {
    return res.status(401).json({ 
      error: 'Unauthorized',
      message: 'Invalid API key'
    });
  }

  next();
};

// Middleware
app.use(helmet());
app.use(compression());
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Health check endpoint
app.get('/health', (_req: Request, res: Response) => {
  res.json({ 
    status: 'ok', 
    initialized: embeddingService.initialized,
    timestamp: new Date().toISOString()
  });
});

// Models endpoint - OpenAI compatible
app.get('/models', (_req: Request, res: Response) => {
  const modelsResponse: ModelsResponse = {
    object: 'list',
    data: [embeddingService.getModelInfo()]
  };
  res.json(modelsResponse);
});

app.get('/v1/models', (_req: Request, res: Response) => {
  const modelsResponse: ModelsResponse = {
    object: 'list',
    data: [embeddingService.getModelInfo()]
  };
  res.json(modelsResponse);
});

// OpenAI-compatible embeddings endpoint
app.post('/v1/embeddings', authenticateApiKey, async (req: Request<Record<string, never>, unknown, EmbeddingsRequest>, res: Response) => {
  const requestStart = Date.now();
  console.time('v1/embeddings request');
  try {
    const { input, model, dimensions, encoding_format = 'float' } = req.body;
    
    if (!input) {
      return res.status(400).json({ 
        error: {
          type: 'invalid_request_error',
          message: 'Missing required parameter: input'
        }
      });
    }

    if (!model) {
      return res.status(400).json({ 
        error: {
          type: 'invalid_request_error',
          message: 'Missing required parameter: model'
        }
      });
    }

    // Validate model
    const modelInfo = embeddingService.getModelInfo();
    if (model !== modelInfo.id) {
      return res.status(400).json({ 
        error: {
          type: 'invalid_request_error',
          message: `Model '${model}' not found. Available models: ${modelInfo.id}`
        }
      });
    }

    // Validate dimensions - for MRL models, recommend specific dimensions
    if (dimensions) {
      if (dimensions < 1 || dimensions > embeddingService.getMaxDimensions()) {
        return res.status(400).json({ 
          error: {
            type: 'invalid_request_error',
            message: `Dimensions must be between 1 and ${embeddingService.getMaxDimensions()}. Recommended MRL-trained dimensions: 768 (full), 512, 256, or 128`
          }
        });
      }
      
      // Warn if non-MRL-trained dimensions are used
      const supportedDims = embeddingService.getSupportedDimensions();
      if (!supportedDims.includes(dimensions)) {
        console.warn(`⚠️  Custom dimension ${dimensions} requested. MRL-trained dimensions for best quality: ${supportedDims.join(', ')}`);
      }
    }

    // Validate encoding format
    if (encoding_format !== 'float') {
      return res.status(400).json({ 
        error: {
          type: 'invalid_request_error',
          message: 'Only "float" encoding format is currently supported'
        }
      });
    }

    // Handle both single string and array of strings
    const texts = Array.isArray(input) ? input : [input];
    
    if (texts.length === 0) {
      return res.status(400).json({ 
        error: {
          type: 'invalid_request_error',
          message: 'Input array cannot be empty'
        }
      });
    }

    const embeddings = await embeddingService.getEmbeddings(texts, dimensions);
    const totalTokens = embeddingService.getTotalTokens(texts);
    
    const embeddingData: EmbeddingData[] = embeddings.map((embedding, index) => ({
      object: 'embedding',
      index,
      embedding
    }));

    const response: EmbeddingsResponse = {
      object: 'list',
      data: embeddingData,
      model: model,
      usage: {
        prompt_tokens: totalTokens,
        total_tokens: totalTokens
      }
    };
    
    const requestDuration = Date.now() - requestStart;
    console.timeEnd('v1/embeddings request');
    console.log(`v1/embeddings request completed in ${requestDuration}ms (${texts.length} texts)`);
    
    res.json(response);
    
  } catch (error) {
    console.timeEnd('v1/embeddings request');
    console.error('Embedding error:', error);
    res.status(500).json({ 
      error: {
        type: 'internal_server_error',
        message: error instanceof Error ? error.message : 'Unknown error'
      }
    });
  }
});

// Legacy embedding endpoint for backward compatibility
app.post('/embed', authenticateApiKey, async (req: Request, res: Response) => {
  const requestStart = Date.now();
  console.time('embed request');
  try {
    const { input, dimensions } = req.body;
    
    if (!input) {
      return res.status(400).json({ 
        error: 'Missing required field: input' 
      });
    }

    // Handle both single string and array of strings
    const texts = Array.isArray(input) ? input : [input];
    
    if (texts.length === 0) {
      return res.status(400).json({ 
        error: 'Input array cannot be empty' 
      });
    }

    const embeddings = await embeddingService.getEmbeddings(texts, dimensions);
    
    const requestDuration = Date.now() - requestStart;
    console.timeEnd('embed request');
    console.log(`embed request completed in ${requestDuration}ms (${texts.length} texts)`);
    
    res.json({
      embeddings: embeddings,
      count: embeddings.length
    });
    
  } catch (error) {
    console.timeEnd('embed request');
    console.error('Embedding error:', error);
    res.status(500).json({ 
      error: 'Failed to generate embeddings',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Reranking endpoint
app.post('/rerank', authenticateApiKey, async (req: Request<Record<string, never>, unknown, RerankRequest>, res: Response) => {
  const requestStart = Date.now();
  console.time('rerank request');
  try {
    const { query, documents } = req.body;
    
    if (!query || !documents) {
      return res.status(400).json({ 
        error: 'Missing required fields: query and documents' 
      });
    }

    if (!Array.isArray(documents)) {
      return res.status(400).json({ 
        error: 'Documents must be an array' 
      });
    }

    if (documents.length === 0) {
      return res.status(400).json({ 
        error: 'Documents array cannot be empty' 
      });
    }

    const ranking = await embeddingService.rerank(query, documents);
    
    const requestDuration = Date.now() - requestStart;
    console.timeEnd('rerank request');
    console.log(`rerank request completed in ${requestDuration}ms (${documents.length} documents)`);
    
    res.json({
      ranking: ranking
    });
    
  } catch (error) {
    console.timeEnd('rerank request');
    console.error('Reranking error:', error);
    res.status(500).json({ 
      error: 'Failed to rerank documents',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Error handling middleware
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  console.error(err.stack);
  res.status(500).json({ 
    error: 'Internal server error',
    message: err.message 
  });
});

// 404 handler
app.use((_req: Request, res: Response) => {
  res.status(404).json({ 
    error: 'Endpoint not found',
    availableEndpoints: ['/health', '/models', '/v1/models', '/v1/embeddings', '/embed', '/rerank']
  });
});

// Start server with SSL fallback
async function startServer(): Promise<void> {
  try {
    console.log('Starting Gemma embedding service...');
    console.time('Server startup');
    
    // Pre-initialize the model
    await embeddingService.initialize();
    
    if (USE_SSL) {
      try {
        // Check if SSL certificates exist
        if (!fs.existsSync(SSL_KEY_PATH) || !fs.existsSync(SSL_CERT_PATH)) {
          console.warn('⚠️  SSL certificates not found:');
          console.warn(`    Private key: ${SSL_KEY_PATH}`);
          console.warn(`    Certificate: ${SSL_CERT_PATH}`);
          console.warn('⚠️  Falling back to HTTP mode...');
          console.warn('    To generate SSL certificates, run: ./generate-ssl-certs.sh');
          throw new Error('SSL certificates not found');
        }

        // Read SSL certificates
        const sslOptions = {
          key: fs.readFileSync(SSL_KEY_PATH),
          cert: fs.readFileSync(SSL_CERT_PATH)
        };

        // Create HTTPS server
        const httpsServer = https.createServer(sslOptions, app);
        
        httpsServer.listen(PORT, () => {
          console.timeEnd('Server startup');
          console.log(`🔒 Gemma embedding service running on HTTPS port ${PORT}`);
          console.log(`Available endpoints:`);
          console.log(`  GET  https://localhost:${PORT}/health - Health check (no auth required)`);
          console.log(`  GET  https://localhost:${PORT}/models - List available models (no auth required)`);
          console.log(`  GET  https://localhost:${PORT}/v1/models - List models (OpenAI compatible, no auth required)`);
          console.log(`  POST https://localhost:${PORT}/v1/embeddings - Generate embeddings (OpenAI compatible, requires API key)`);
          console.log(`  POST https://localhost:${PORT}/embed - Generate embeddings (legacy, requires API key)`);
          console.log(`  POST https://localhost:${PORT}/rerank - Rerank documents (requires API key)`);
          console.log(`🔐 Authentication: Set API_KEY environment variable`);
        });

        console.log('✅ SSL/HTTPS mode enabled successfully');
        return;

      } catch (sslError) {
        console.warn('⚠️  Failed to start HTTPS server:', sslError instanceof Error ? sslError.message : sslError);
        console.warn('⚠️  Falling back to HTTP mode...');
      }
    }

    // Create HTTP server (fallback or when USE_SSL=false)
    app.listen(PORT, () => {
      console.timeEnd('Server startup');
      console.log(`🌐 Gemma embedding service running on HTTP port ${PORT}`);
      console.log(`Available endpoints:`);
      console.log(`  GET  http://localhost:${PORT}/health - Health check (no auth required)`);
      console.log(`  GET  http://localhost:${PORT}/models - List available models (no auth required)`);
      console.log(`  GET  http://localhost:${PORT}/v1/models - List models (OpenAI compatible, no auth required)`);
      console.log(`  POST http://localhost:${PORT}/v1/embeddings - Generate embeddings (OpenAI compatible, requires API key)`);
      console.log(`  POST http://localhost:${PORT}/embed - Generate embeddings (legacy, requires API key)`);
      console.log(`  POST http://localhost:${PORT}/rerank - Rerank documents (requires API key)`);
      console.log(`🔐 Authentication: Set API_KEY environment variable`);
      
      if (USE_SSL) {
        console.warn('⚠️  Running in HTTP mode due to SSL setup failure');
        console.warn('    To enable HTTPS:');
        console.warn('    1. Run ./generate-ssl-certs.sh to create certificates');
        console.warn('    2. Or provide custom certificates via SSL_KEY_PATH and SSL_CERT_PATH');
      } else {
        console.log('ℹ️  HTTP mode (set USE_SSL=true to enable HTTPS)');
      }
    });

  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('Received SIGTERM, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('Received SIGINT, shutting down gracefully');
  process.exit(0);
});

if (require.main === module) {
  startServer();
}

export default app;
