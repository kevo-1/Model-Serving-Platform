package main

import (
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"runtime"

	"github.com/joho/godotenv"
	httpHandler "github.com/kevo-1/model-serving-platform/internal/handler/http"
	"github.com/kevo-1/model-serving-platform/internal/repository"
	"github.com/kevo-1/model-serving-platform/pkg/onnx"
	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	if err := godotenv.Load(); err != nil {
        log.Println("No .env file found, using system environment")
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
			log.Fatal("Unsupported operating system")
		}
	}

	log.Printf("Using ONNX library path: %s", libraryPath)

	ort.SetSharedLibraryPath(libraryPath)

	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("Failed to initialize ONNX environment: %v", err)
	}
    defer ort.DestroyEnvironment()

    // Step 1: Create model registry
	registry := repository.NewModelRegistery()
	
	
    // Step 2: Load dummy model(s)
	dummyModel := onnx.NewDummyPredictor(
		"iris_v1",              
		"Iris Classifier",      
		"v1.0.0",              
		"models/iris_classifier_v1.onnx",
	)

	if err := registry.Register("iris_v1", dummyModel); err != nil {
		log.Fatalf("Failed to register model: %v", err)
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

	log.Printf("Server starting on port %s", port)
	log.Printf("Available endpoints:")
	log.Printf("  POST http://localhost:%s/predict", port)
	log.Printf("  GET  http://localhost:%s/health", port)
	log.Printf("  GET  http://localhost:%s/models", port)

    
    // Step 6: Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		if err := http.ListenAndServe(addr, routes); err != nil {
			log.Fatalf("Server failed: %v", err)
		}
	}()

	log.Println("Server running. Press Ctrl+C to stop.")

	<-sigChan
	log.Println("Shutting down server...")
}