data "aws_caller_identity" "me" {}

locals {
  name = "aws-ai-lab"
}

# ---------------------------------------------------------------- image registry
resource "aws_ecr_repository" "app" {
  name         = local.name
  force_delete = true # lab: let `terraform destroy` remove images too
}

# Keep only the 3 newest images (ECR storage is $0.10/GB-month; keeps it ~$0)
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep last 3"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 3 }
      action       = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------- IAM (least privilege)
data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "fn" {
  name               = "${local.name}-fn"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

data "aws_iam_policy_document" "fn" {
  # Claude via inference profile needs BOTH the profile ARN and the underlying foundation-model ARNs
  # (the profile fans out to several regions, hence the wildcard region).
  statement {
    actions = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
    resources = [
      "arn:aws:bedrock:*::foundation-model/anthropic.*",
      "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0",
      "arn:aws:bedrock:*:${data.aws_caller_identity.me.account_id}:inference-profile/*anthropic.*",
    ]
  }
}

resource "aws_iam_role_policy" "fn" {
  role   = aws_iam_role.fn.id
  policy = data.aws_iam_policy_document.fn.json
}

resource "aws_iam_role_policy_attachment" "logs" {
  role       = aws_iam_role.fn.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ---------------------------------------------------------------- lambda + function url
resource "aws_cloudwatch_log_group" "fn" {
  name              = "/aws/lambda/${local.name}"
  retention_in_days = 7
}

resource "aws_lambda_function" "app" {
  count         = var.image_tag == "" ? 0 : 1
  function_name = local.name
  role          = aws_iam_role.fn.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
  memory_size   = 1024
  timeout       = 60

  environment {
    variables = {
      BEDROCK_CHAT_MODEL = var.chat_model
      EMBED_DIMS         = tostring(var.embed_dims)
    }
  }
  depends_on = [aws_cloudwatch_log_group.fn, aws_iam_role_policy.fn]
}

# IAM-authenticated URL: AWS rejects unsigned requests before the function runs, so strangers
# can't spend your tokens. Callers sign with SigV4 (e.g. curl --aws-sigv4). Same-account IAM
# principals with lambda:InvokeFunctionUrl (e.g. your admin user) need no resource policy.
resource "aws_lambda_function_url" "app" {
  count              = var.image_tag == "" ? 0 : 1
  function_name      = aws_lambda_function.app[0].function_name
  authorization_type = "AWS_IAM"
  invoke_mode        = "RESPONSE_STREAM" # SSE token streaming
}

# ---------------------------------------------------------------- cost guardrail
resource "aws_budgets_budget" "monthly" {
  name         = "${local.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }
}
