package logger

import (
	"log/slog"
	"os"
)

var logger *slog.Logger


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