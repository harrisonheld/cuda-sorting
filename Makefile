CUDA_PATH       := /usr
NVCC            := $(CUDA_PATH)/bin/nvcc
SRC_DIR         := src
BIN_DIR         := bin

TARGET          := $(BIN_DIR)/vector_add
SRC             := $(SRC_DIR)/vector_add.cu

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(SRC) -o $(TARGET)

clean:
	rm -rf $(BIN_DIR)

run: all
	./$(TARGET)
