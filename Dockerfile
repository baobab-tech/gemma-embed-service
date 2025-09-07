# Use Node.js 24 Alpine 3.22 image for smaller size
FROM node:24-alpine3.22

# Set working directory
WORKDIR /app

# Install Python and build dependencies for native modules
RUN apk add --no-cache python3 make g++ 

# Copy package files first for better caching
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci --only=production

# Install development dependencies for build
RUN npm install --only=dev

# Copy source code
COPY src ./src

# Build TypeScript
RUN npm run build

# Remove dev dependencies after build
RUN npm prune --production

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

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
CMD ["npm", "start"]
