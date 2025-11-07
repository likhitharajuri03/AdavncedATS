FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install --silent

# Copy source
COPY . /app

EXPOSE 3000

# Start the frontend dev server (React)
CMD ["npm", "start"]

