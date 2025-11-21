# Setup
```
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

# Running
```
make run
```

Original array: 6 5 4 2 1 0 , pivot = 3
num_blocks=3
below_arr: 0 1 2 
above_arr: 2 1 0 
offset_below: 0 0 1 
offset_above: 3 5 6 
total_below = 3
Array after reordering: 2 1 0 6 5 4

4096,"Quick Sort",0.143364,PASS
16384,"Quick Sort",0.0163028,PASS
65536,"Quick Sort",0.0702595,PASS
262144,"Quick Sort",0.398046,PASS
1048576,"Quick Sort",1.14716,PASS