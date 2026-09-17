# Production Nginx image for NewsLens-AI Frontend
FROM nginx:1.27-alpine AS runner

RUN apk add --no-cache gettext

COPY dist /usr/share/nginx/html
COPY nginx.conf.template /etc/nginx/conf.d/default.conf.template
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENV PORT=8080
ENV BACKEND_SERVICE_URL=http://backend:8000

EXPOSE 8080 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:${PORT}/_health || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
