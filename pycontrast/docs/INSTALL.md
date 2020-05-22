### Prerequisites
- Lnux
- Python3
- NVIDIA GPUs + CUDA CuDNN

### Preparation
You can set up the environments in your own way (`requirements.txt` has been provided), or following below steps: 
1. Clone this repo with:
	```
	git clone https://github.com/HobbitLong/PyContrast.git
	cd PyContrast/pycontrast
	```
2. (Optional) you can consider setting up a virtual environment. Such environment 
    (e.g. `PyConEnv`) can be created by
	```
	virtualenv -p python3 ~/env/PyConEnv
	```
    then activate it:
    ```
	source ~/env/PyConEnv/bin/activate
	```
3. Install packages:
	```
	pip3 install -r requirements.txt
	```
4. (Optional) install [apex](https://github.com/NVIDIA/apex) if you would like to 
try mixed precision training. If you do not want to take a look at the 
[apex](https://github.com/NVIDIA/apex) repo, the installing commands are (assuming pytorch 
and CUDA are availabel):
    ```
    cd ~
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
	```