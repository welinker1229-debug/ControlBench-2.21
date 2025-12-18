# IMAGE_NAME := zjjjjj1905/controbench
IMAGE_NAME := controbench
CUDA_TAG   ?= cu124
CPU_TAG    ?= cpu

build:
	docker build -t $(IMAGE_NAME):$(CUDA_TAG) \
		--build-arg DGL_WHEEL_SRC=https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html \
		--build-arg DGL_PACKAGE=dgl \
		--build-arg TORCH_VERSION=2.4.0 .
		
build-no-cuda:
	docker build -t $(IMAGE_NAME):$(CPU_TAG) \
		--build-arg DGL_PACKAGE=dgl==1.1.3 \
		--build-arg TORCH_VERSION=2.0.1 .

bash:
	docker run --gpus all -it --rm \
		-v "$(CURDIR)":/app \
		-w /app \
		$(IMAGE_NAME):$(CUDA_TAG) bash

bash-no-cuda:
	docker run -it --rm \
		-v "$(CURDIR)":/app \
		-w /app \
		$(IMAGE_NAME):$(CPU_TAG) bash

clean:
	-docker image rm -f $(IMAGE_NAME):$(CUDA_TAG) $(IMAGE_NAME):$(CPU_TAG)
