
update: 
	git reset --hard
	git pull

setup:
	pip install -r requirements.txt
	@mkdir -p ./output/validation
	@mkdir -p ./data

sample-test: \
	setup \
	sample-clean \
	sample-validate \
	sample-train
	$(shell pwd)/bin/train \
		--mode=test \
		--config-dir=config/samples/20231212_transformed \
		--target-config-name=train_X_231211_final_col_01_transformed \
		--target-model-path=output/sample-model.pth

sample-clean:
	@rm -f output/sample-model.pth

sample-train:
	$(shell pwd)/bin/train \
		--mode=train \
		--config-dir=config/samples/20231212_transformed \
		--output-model-name=sample-model.pth

sample-validate:
	$(shell pwd)/bin/train --mode=validate --config-dir=config/samples/20231212_transformed


data-dir-load-to-server:
	scp -r $(shell pwd)/data/* estsoft@45.32.129.17:/data

server-output-to-local:
	scp -r estsoft@45.32.129.17:/output/* $(shell pwd)/output

require-redis: 
	docker run \
	--rm \
	--detach \
	--name redis \
	--publish 6379:6379 \
	redis