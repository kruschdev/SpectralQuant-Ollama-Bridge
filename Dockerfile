FROM node:22-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY index.js .

# Expose bridge port
EXPOSE 11437

# Run the proxy
CMD ["node", "index.js"]
