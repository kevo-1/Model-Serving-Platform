package http

import (
    "net/http"
    "context"
    "time"
    
    "github.com/google/uuid" 
    "github.com/kevo-1/model-serving-platform/internal/logger"
    "github.com/kevo-1/model-serving-platform/internal/metrics"
)


func RequestIDMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        requestID := uuid.New().String()

        ctx := context.WithValue(r.Context(), logger.RequestIDKey, requestID)
        r = r.WithContext(ctx)

        w.Header().Set("X-Request-ID", requestID)

        logger.Info("http request received", 
            "request_id", requestID,
            "method", r.Method,
            "path", r.URL.Path,
        )

        next.ServeHTTP(w, r)
    })
}


type responseWriter struct {
    http.ResponseWriter
    statusCode int
}

func newResponseWriter(w http.ResponseWriter) *responseWriter {
	return &responseWriter{w, http.StatusOK}
}

func (rw *responseWriter) WriteHeader(code int) {
    rw.statusCode = code
    rw.ResponseWriter.WriteHeader(code)
}

func MetricsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
            next.ServeHTTP(w, r)
            return
        }
		
        start := time.Now()
        
        wrapped := newResponseWriter(w)
        
        next.ServeHTTP(wrapped, r)

        duration := time.Since(start).Seconds()
        metrics.RecordHTTPRequest(r.Method, r.URL.Path, wrapped.statusCode, duration)
    })
}