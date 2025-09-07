# Use Node.js 24 Debian slim image for glibc compatibility
FROM node:24-slim

# Set working directory
WORKDIR /app

# Install Python and build dependencies for native modules
RUN apt-get update && apt-get install -y \
    python3 \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/* 

# Install pnpm globally
RUN npm install -g pnpm

# Copy package files first for better caching
COPY package*.json pnpm-lock.yaml ./
COPY tsconfig.json ./

# Install all dependencies (production and dev for build)
RUN pnpm install --frozen-lockfile

# Copy source code and SSL generation script
COPY src ./src
COPY generate-ssl-certs.sh ./

# Build TypeScript
RUN pnpm run build

# Remove dev dependencies after build
RUN pnpm prune --production

# Create certs directory and make SSL script executable
RUN mkdir -p certs && chmod +x generate-ssl-certs.sh

# Create non-root user for security
RUN groupadd --gid 1001 nodejs && \
    useradd --uid 1001 --gid 1001 --system --create-home nodejs

# Change ownership of app directory
RUN chown -R nodejs:nodejs /app

# Switch to non-root user
USER nodejs

# Expose port
EXPOSE 8082

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "const http = require('http'); \
    const options = { hostname: 'localhost', port: 8082, path: '/health', timeout: 2000 }; \
    const req = http.request(options, (res) => { process.exit(res.statusCode === 200 ? 0 : 1); }); \
    req.on('error', () => process.exit(1)); \
    req.end();"

# Start the application
CMD ["pnpm", "start"]
