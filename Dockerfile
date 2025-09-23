# Build the static assets with TailwindCSS.
FROM node:22-alpine AS build
WORKDIR /app

# Install dependencies (including Tailwind CLI) needed to build the CSS bundle.
COPY package*.json ./
RUN npm ci

# Copy only the files required for the Tailwind build to keep the context small.
COPY tailwind.config.cjs ./
COPY web ./web
RUN npm run build:tailwind

# Use BusyBox for an ultra-lean runtime that only serves static assets.
FROM busybox:1.36
WORKDIR /www

# Copy the pre-built site into the runtime image.
COPY --from=build /app/web ./

EXPOSE 8080
CMD ["httpd", "-f", "-p", "8080", "-h", "/www"]
