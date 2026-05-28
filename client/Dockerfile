FROM node:22-alpine AS builder

WORKDIR /app

RUN corepack enable

# Copiar solo archivos de configuración primero (caché de capas)
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml* ./
RUN pnpm install --frozen-lockfile --ignore-scripts

# Copiar el resto del código y compilar
COPY . .
RUN pnpm run build

FROM node:22-alpine AS runner
WORKDIR /app

RUN corepack enable

COPY --from=builder /app/package.json /app/pnpm-lock.yaml ./
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/node_modules ./node_modules

EXPOSE 3000

CMD ["pnpm", "start"]