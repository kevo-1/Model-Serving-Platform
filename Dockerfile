# Build stage
FROM golang:1.24 AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -o server ./cmd/server


# Final stage
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz \
    && tar -xzf onnxruntime-linux-x64-1.16.3.tgz \
    && cp onnxruntime-linux-x64-1.16.3/lib/libonnxruntime.so.1.16.3 /usr/lib/libonnxruntime.so \
    && rm -rf onnxruntime-linux-x64-1.16.3*


WORKDIR /app

COPY --from=builder /app/server .

COPY models ./models
COPY index.html .

ENV LD_LIBRARY_PATH=/usr/lib

EXPOSE 8080

CMD ["./server"]
