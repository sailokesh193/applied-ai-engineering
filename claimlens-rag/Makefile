.PHONY: install test eval run docker aws-push azure-push

install:
	pip install -e ".[dev,aws,azure,openai]"

test:
	pytest -q

eval:
	claimlens ingest data/sample
	claimlens eval evals/golden.jsonl --min-hit-rate 0.8 --min-refusal-accuracy 1.0

run:
	claimlens serve --port 8000

docker:
	docker build -t claimlens:latest .

# make aws-push AWS_REGION=us-east-1 ECR=<terraform output ecr_repository_url>
aws-push: docker
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(firstword $(subst /, ,$(ECR)))
	docker tag claimlens:latest $(ECR):latest
	docker push $(ECR):latest

# make azure-push ACR=<terraform output acr_login_server>
azure-push:
	az acr build --registry $(firstword $(subst ., ,$(ACR))) --image claimlens:latest .
