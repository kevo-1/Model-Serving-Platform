package main

import (
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"syscall"

	"github.com/joho/godotenv"
	httpHandler "github.com/kevo-1/model-serving-platform/internal/handler/http"
	"github.com/kevo-1/model-serving-platform/internal/logger"
	"github.com/kevo-1/model-serving-platform/internal/repository"
	"github.com/kevo-1/model-serving-platform/pkg/onnx"
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
    defer ort.DestroyEnvironment()

    // Step 1: Create model registry
	registry := repository.NewModelRegistery()
	
	
    // Step 2: Load model(s)
	onnxModel, err := onnx.NewONNXPredictor(
		"iris_v1",
		"Iris Classifier",
		"v1.0.0",
		"models/iris_classifier_v1.onnx",
	)
	if err != nil {
		logger.Error("failed to load onnx model", "error", err)
		os.Exit(1)
	}
	defer onnxModel.Close()
	
	if err := registry.Register("iris_v1", onnxModel); err != nil {
		logger.Error("failed to register model", "error", err)
		os.Exit(1)
	}
    
    // Step 3: Create HTTP handler
	handler := httpHandler.NewHandler(registry)
	routes := handler.SetupRoutes()

	
    // Step 4: Setup HTTP server
	
    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }
	addr := ":" + port

	logger.Info("server starting", "port", port)
	logger.Info("endpoints available",
		"predict", fmt.Sprintf("POST http://localhost:%s/predict", port),
		"health", fmt.Sprintf("GET http://localhost:%s/health", port),
		"models", fmt.Sprintf("GET http://localhost:%s/models", port),
		"metrics", fmt.Sprintf("GET http://localhost:%s/metrics", port),
	)

    // Step 6: Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		if err := http.ListenAndServe(addr, routes); err != nil {
			logger.Error("server failed", "error", err)
		}
	}()

	logger.Info("Server running. Press Ctrl+C to stop.")
	
	<-sigChan
	logger.Info("Shutting down server...")
}