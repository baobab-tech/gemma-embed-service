import express, { type Request, type Response, type NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import https from 'https';
import fs from 'fs';
import path from 'path';
import { AutoModel, AutoTokenizer, matmul } from '@huggingface/transformers';

interface EmbedRequest {
  input: string | string[];
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
    const model_id = "onnx-community/embeddinggemma-300m-ONNX";
    
    try {
      this.tokenizer = await AutoTokenizer.from_pretrained(model_id);
      this.model = await AutoModel.from_pretrained(model_id, {
        dtype: "fp32",
      });
      
      this.isInitialized = true;
      console.log('Model initialized successfully');
    } catch (error) {
      console.error('Failed to initialize model:', error);
      throw error;
    }
  }

  async getEmbeddings(texts: string[]): Promise<number[][]> {
    await this.initialize();
    
    const processedTexts = texts.map(text => this.prefixes.document + text);
    const inputs = await (this.tokenizer as any)(processedTexts, { padding: true });
    const { sentence_embedding } = await (this.model as any)(inputs);
    
    return sentence_embedding.tolist();
  }

  async rerank(query: string, documents: string[]): Promise<number[]> {
    await this.initialize();
    
    const queryText = this.prefixes.query + query;
    const documentTexts = documents.map(doc => this.prefixes.document + doc);
    const allTexts = [queryText, ...documentTexts];
    
    const inputs = await (this.tokenizer as any)(allTexts, { padding: true });
    const { sentence_embedding } = await (this.model as any)(inputs);
    
    // Compute similarities
    const scores = await matmul(sentence_embedding, sentence_embedding.transpose(1, 0));
    const similarities: number[] = scores.tolist()[0].slice(1);
    
    // Create ranking with original indices
    const ranking: RankingItem[] = similarities
      .map((score, index) => ({ index, score }))
      .sort((a, b) => b.score - a.score);
    
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

// Embedding endpoint
app.post('/embed', async (req: Request<Record<string, never>, unknown, EmbedRequest>, res: Response) => {
  try {
    const { input } = req.body;
    
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

    const embeddings = await embeddingService.getEmbeddings(texts);
    
    res.json({
      embeddings: embeddings,
      count: embeddings.length
    });
    
  } catch (error) {
    console.error('Embedding error:', error);
    res.status(500).json({ 
      error: 'Failed to generate embeddings',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Reranking endpoint
app.post('/rerank', async (req: Request<Record<string, never>, unknown, RerankRequest>, res: Response) => {
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
    
    res.json({
      ranking: ranking
    });
    
  } catch (error) {
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
    availableEndpoints: ['/health', '/embed', '/rerank']
  });
});

// Start server with SSL fallback
async function startServer(): Promise<void> {
  try {
    console.log('Starting Gemma embedding service...');
    
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
          console.log(`🔒 Gemma embedding service running on HTTPS port ${PORT}`);
          console.log(`Available endpoints:`);
          console.log(`  GET  https://localhost:${PORT}/health - Health check`);
          console.log(`  POST https://localhost:${PORT}/embed - Generate embeddings`);
          console.log(`  POST https://localhost:${PORT}/rerank - Rerank documents`);
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
      console.log(`🌐 Gemma embedding service running on HTTP port ${PORT}`);
      console.log(`Available endpoints:`);
      console.log(`  GET  http://localhost:${PORT}/health - Health check`);
      console.log(`  POST http://localhost:${PORT}/embed - Generate embeddings`);
      console.log(`  POST http://localhost:${PORT}/rerank - Rerank documents`);
      
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
