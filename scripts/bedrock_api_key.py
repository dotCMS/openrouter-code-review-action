"""Print a short-term Amazon Bedrock API key minted from the job's AWS credentials.

The key is derived from the role session that aws-actions/configure-aws-credentials
set up, and it expires with that session. The action masks it before use and passes
only the key, never the role credentials, to the review CLI.
"""

import os

from aws_bedrock_token_generator import provide_token

print(provide_token(region=os.environ.get("AWS_REGION") or "us-east-1"))
