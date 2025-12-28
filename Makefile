# IMAGE_NAME := zjjjjj1905/controbench
IMAGE_NAME := controbenchv3

build:
	docker build --no-cache -t $(IMAGE_NAME):gpu \
		--build-arg DGL_WHEEL_SRC=https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html \
		--build-arg DGL_PACKAGE=dgl \
		--build-arg TORCH_VERSION=2.4.0 \
		.
		
build-no-cuda:
	docker build --no-cache -t $(IMAGE_NAME):cpu \
		--build-arg TORCH_VERSION=2.4.0 \
		--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
		--build-arg DGL_PACKAGE=dgl==2.4.0 \
		--build-arg DGL_WHEEL_SRC=https://data.dgl.ai/wheels/torch-2.4/repo.html \
		--build-arg PYG_WHEEL_SRC=https://data.pyg.org/whl/torch-2.4.0+cpu.html \
		.

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
	-docker image rm -f $(IMAGE_NAME):gpu $(IMAGE_NAME):cpu
