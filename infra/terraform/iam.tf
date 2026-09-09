data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${var.service_name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# The task role reads model artifacts and nothing else - the serving container has
# no reason to be able to write to the model bucket.
resource "aws_iam_role" "task" {
  name               = "${var.service_name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

data "aws_iam_policy_document" "task" {
  statement {
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [var.model_bucket_arn, "${var.model_bucket_arn}/*"]
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "${var.service_name}-task-policy"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}
