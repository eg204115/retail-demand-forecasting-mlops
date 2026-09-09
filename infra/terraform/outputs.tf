output "ecr_repository_url" {
  value = aws_ecr_repository.serving.repository_url
}

output "cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "service_name" {
  value = aws_ecs_service.serving.name
}

output "log_group" {
  value = aws_cloudwatch_log_group.serving.name
}
