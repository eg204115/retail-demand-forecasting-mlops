# Serving infrastructure: ECR for the image, Fargate for the service.
# Deliberately small - the interesting engineering is in the pipeline, and this
# exists so the deployment is reproducible rather than clicked together in a console.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
  }
  # Remote state so CI and a laptop cannot both own the same infrastructure.
  backend "s3" {
    key = "shelfcast/serving.tfstate"
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "shelfcast"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

resource "aws_ecr_repository" "serving" {
  name                 = var.ecr_repository
  image_tag_mutability = "IMMUTABLE" # a tag must always mean the same image
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "serving" {
  repository = aws_ecr_repository.serving.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 20 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 20 }
      action       = { type = "expire" }
    }]
  })
}

resource "aws_ecs_cluster" "this" {
  name = var.cluster_name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_log_group" "serving" {
  name              = "/ecs/${var.service_name}"
  retention_in_days = 30
}

resource "aws_ecs_task_definition" "serving" {
  family                   = var.service_name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name         = "serving"
    image        = "${aws_ecr_repository.serving.repository_url}:${var.image_tag}"
    essential    = true
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    environment = [
      { name = "MODEL_NAME", value = var.model_name },
      { name = "MODEL_STAGE", value = var.model_stage },
      { name = "MLFLOW_TRACKING_URI", value = var.mlflow_tracking_uri },
      { name = "REDIS_HOST", value = var.redis_host },
      { name = "LOG_LEVEL", value = "INFO" },
    ]
    # The deep check: an unhealthy task is one that cannot score, not one whose
    # process happens to be alive.
    healthCheck = {
      command     = ["CMD-SHELL", "curl -fsS http://localhost:8000/ready || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.serving.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

resource "aws_ecs_service" "serving" {
  name            = var.service_name
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.serving.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  # Rolling deploy with a circuit breaker, so a bad image rolls itself back.
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  deployment_maximum_percent         = 200
  deployment_minimum_healthy_percent = 100

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.serving.id]
    assign_public_ip = false
  }

  lifecycle {
    # CI updates the image; Terraform should not fight it back to the pinned tag.
    ignore_changes = [task_definition]
  }
}

resource "aws_appautoscaling_target" "serving" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.serving.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.desired_count
  max_capacity       = var.max_capacity
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "${var.service_name}-cpu"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.serving.service_namespace
  resource_id        = aws_appautoscaling_target.serving.resource_id
  scalable_dimension = aws_appautoscaling_target.serving.scalable_dimension

  target_tracking_scaling_policy_configuration {
    target_value = 65
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}

resource "aws_security_group" "serving" {
  name        = "${var.service_name}-sg"
  description = "Shelfcast serving tasks"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = var.ingress_security_group_ids
    description     = "ALB to task"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
