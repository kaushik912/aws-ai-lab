output "ecr_repo_url" {
  value = aws_ecr_repository.app.repository_url
}

output "function_url" {
  value = try(aws_lambda_function_url.app[0].function_url, null)
}
