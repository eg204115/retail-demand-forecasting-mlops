variable "aws_region" {
  type    = string
  default = "ap-south-1"
}

variable "environment" {
  type    = string
  default = "staging"
}

variable "ecr_repository" {
  type    = string
  default = "shelfcast-serving"
}

variable "cluster_name" {
  type    = string
  default = "shelfcast"
}

variable "service_name" {
  type    = string
  default = "shelfcast-serving"
}

variable "image_tag" {
  type        = string
  description = "Commit SHA of the image to run"
}

variable "task_cpu" {
  type    = number
  default = 1024
}

variable "task_memory" {
  type    = number
  default = 2048
}

variable "desired_count" {
  type    = number
  default = 2
}

variable "max_capacity" {
  type    = number
  default = 8
}

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  type = list(string)
}

variable "ingress_security_group_ids" {
  type    = list(string)
  default = []
}

# Base registered-model name; the container derives <name>-q50 and <name>-q90.
variable "model_name" {
  type    = string
  default = "shelfcast-demand-forecaster"
}

variable "model_stage" {
  type    = string
  default = "Production"
}

variable "mlflow_tracking_uri" {
  type = string
}

variable "model_bucket_arn" {
  type = string
}

variable "redis_host" {
  type = string
}
