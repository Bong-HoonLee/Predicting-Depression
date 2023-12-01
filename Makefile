
update: 
	git pull


training-data-load-to-server:
	scp -r $(shell pwd)/data/* estsoft@45.32.129.17:~/data


require-redis: 
	docker run \
	--rm \
	--detach \
	--name redis \
	--publish 6379:6379 \
	redis