package logger

import (
	"log/slog"
	"context"
	"os"
)

var logger *slog.Logger

type contextKey string
const RequestIDKey contextKey = "request_id"

func GetRequestID(ctx context.Context) string {
    if id, ok := ctx.Value(RequestIDKey).(string); ok {
        return id
    }
    return "unknown"
}


func Init() {
	handler := slog.NewJSONHandler(os.Stdout, nil)
	logger = slog.New(handler)
}

func Info(msg string, args ...any) {
	logger.Info(msg, args...)
}

func Error(msg string, args ...any) {
	logger.Error(msg, args...)
}

func Warn(msg string, args ...any) {
	logger.Warn(msg, args...)
}

func Debug(msg string, args ...any) {
	logger.Debug(msg, args...)
}