#!/bin/sh
set -e

export PORT=${PORT:-8080}
export BACKEND_SERVICE_URL=${BACKEND_SERVICE_URL:-http://backend:8000}

# Substitute PORT and BACKEND_SERVICE_URL into default.conf
envsubst '${PORT} ${BACKEND_SERVICE_URL}' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf

exec nginx -g 'daemon off;'
