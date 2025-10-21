NVCC = nvcc
CFLAGS = -O3 -std=c++17

SRC = src
INC = include
BIN = bin
TARGET = $(BIN)/cuda_sort

SOURCES = $(wildcard $(SRC)/*.cu)
OBJECTS = $(SOURCES:$(SRC)/%.cu=$(BIN)/%.o)

all: $(TARGET)

$(BIN):
	mkdir -p $(BIN)

# Compile
$(BIN)/%.o: $(SRC)/%.cu | $(BIN)
	$(NVCC) $(CFLAGS) -I$(INC) -c $< -o $@

# Link
$(TARGET): $(OBJECTS) | $(BIN)
	$(NVCC) $(CFLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BIN)/*
