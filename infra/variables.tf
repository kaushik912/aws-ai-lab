variable "region" {
  type    = string
  default = "us-east-1"
}

variable "chat_model" {
  description = "Bedrock inference profile id for chat"
  type        = string
  default     = "us.anthropic.claude-haiku-5-5"
}

variable "embed_dims" {
  type    = number
  default = 512
}

variable "image_tag" {
  description = "ECR image tag to deploy. Empty = only the ECR repo is useful (bootstrap step 1)."
  type        = string
  default     = ""
}

variable "budget_usd" {
  description = "Monthly budget alarm threshold"
  type        = number
  default     = 10
}

variable "budget_email" {
  description = "Email for budget alerts (put in terraform.tfvars, which is gitignored)"
  type        = string
}
