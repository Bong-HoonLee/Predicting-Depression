
update: 
	git reset --hard
	git pull


train-test:
	$(shell pwd)/bin/train --mode=train --config-dir=config/samples/20231201

validate-test:
	$(shell pwd)/bin/train --mode=validate --config-dir=config/samples/20231201


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