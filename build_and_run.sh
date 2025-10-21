rm -fr bin
mkdir -p bin
nvcc src/vector_add.cu -o bin/vector_add
./bin/vector_add