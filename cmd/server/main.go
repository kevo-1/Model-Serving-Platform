package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/joho/godotenv"
	httpHandler "github.com/kevo-1/model-nexus/internal/handler/http"
	"github.com/kevo-1/model-nexus/internal/logger"
	"github.com/kevo-1/model-nexus/internal/repository"
	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	logger.Init()

	if err := godotenv.Load(); err != nil {
		logger.Info("No .env file found, using system environment")
	}

	libraryPath := os.Getenv("ONNX_LIBRARY_PATH")

	if libraryPath == "" {
		switch runtime.GOOS {
		case "linux":
			libraryPath = "/usr/lib/libonnxruntime.so"
		case "darwin":
			libraryPath = "/usr/lib/libonnxruntime.dylib"
		case "windows":
			libraryPath = "C:\\Program Files\\onnxruntime\\lib\\onnxruntime.dll"
		default:
			logger.Error("Unsupported operating system")
			os.Exit(1)
		}
	}

	logger.Info("using onnx library", "path", libraryPath)

	ort.SetSharedLibraryPath(libraryPath)

	if err := ort.InitializeEnvironment(); err != nil {
		logger.Error("failed to initialize onnx environment", "error", err)
		os.Exit(1)
	}
	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			logger.Error("failed to destroy onnx environment", "error", err)
		}
	}()

	// Step 1: Create model registry
	registry := repository.NewModelRegistry()

	// Step 2: Create HTTP handler (models can be uploaded dynamically via POST /models/upload)
	handler := httpHandler.NewHandler(registry, "models")
	routes := handler.SetupRoutes()

	// Step 3: Setup HTTP server

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	addr := ":" + port

	logger.Info("server starting", "port", port)
	logger.Info("endpoints available",
		"upload", fmt.Sprintf("POST http://localhost:%s/models/upload", port),
		"predict", fmt.Sprintf("POST http://localhost:%s/predict", port),
		"health", fmt.Sprintf("GET http://localhost:%s/health", port),
		"models", fmt.Sprintf("GET http://localhost:%s/models", port),
		"metrics", fmt.Sprintf("GET http://localhost:%s/metrics", port),
	)

	// Step 4: Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt)
	// SIGTERM is not available on Windows; only os.Interrupt is cross-platform
	if runtime.GOOS != "windows" {
		signal.Notify(sigChan, syscall.SIGTERM)
	}

	srv := &http.Server{
		Addr:    addr,
		Handler: routes,
	}

	go func() {
		logger.Info("Server running. Press Ctrl+C to stop.")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("server failed", "error", err)
		}
	}()

	<-sigChan
	logger.Info("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logger.Error("server shutdown failed", "error", err)
	}

	logger.Info("Server stopped gracefully")
}
