FROM node:22-bookworm-slim
WORKDIR /app
COPY package.json ./
COPY server.js ./
COPY lib ./lib
COPY public ./public
COPY config ./config
COPY tests ./tests
RUN mkdir -p /app/data && chown -R node:node /app
USER node
ENV NODE_ENV=production HOST=0.0.0.0 PORT=3000 EXPERIMENT_DATABASE=/app/data/experiment.sqlite3
EXPOSE 3000
CMD ["node", "server.js"]
