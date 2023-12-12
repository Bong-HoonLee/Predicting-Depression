
update: 
	git reset --hard
	git pull


train-test:
	$(shell pwd)/bin/train --mode=train --config-dir=config/samples/20231212_transformed

validate-test:
	$(shell pwd)/bin/train --mode=validate --config-dir=config/samples/20231212_transformed

testset-test:
	$(shell pwd)/bin/train \
		--mode=test \
		--config-dir=config/samples/20231212_transformed \
		--config-name=train_X_231211_final_col_01_transformed \
		--model-path=output/train_X_231211_final_col_01_transformed_202312112151.pth


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