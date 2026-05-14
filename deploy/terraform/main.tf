# Minimal Terraform recipe for ht-repro on AWS EC2 (L40S, g6e.xlarge).
# Bring your own state backend; this file intentionally has no backend block.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

variable "region"             { default = "us-east-1" }
variable "instance_type"      { default = "g6e.xlarge" }   # 1x L40S 48GB
variable "key_name"           { description = "Existing EC2 key pair name" }
variable "allowed_ssh_cidr"   { default = "0.0.0.0/0" }
variable "ht_repro_token"     { sensitive = true }
variable "ami_id" {
  # Ubuntu 22.04 LTS w/ NVIDIA drivers — region-specific; override or use a data lookup
  default = "ami-0c7217cdde317cfec"
}

provider "aws" { region = var.region }

resource "aws_security_group" "ht_repro" {
  name        = "ht-repro-sg"
  description = "ht-repro API server"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port = 0; to_port = 0; protocol = "-1"; cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "ht_repro" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.ht_repro.id]

  root_block_device {
    volume_size = 200
    volume_type = "gp3"
  }

  user_data = <<-EOT
    #!/bin/bash
    set -euxo pipefail
    apt-get update
    apt-get install -y docker.io docker-compose-plugin git nginx
    systemctl enable --now docker

    git clone https://github.com/NagusameCS/HyperTensor.git /opt/HyperTensor
    cd /opt/HyperTensor/deploy
    cat > .env <<EOF
    HT_REPRO_TOKEN=${var.ht_repro_token}
    HT_REPRO_PORT=8765
    EOF
    docker compose up -d --build

    cp nginx.conf /etc/nginx/sites-available/ht-repro
    ln -sf /etc/nginx/sites-available/ht-repro /etc/nginx/sites-enabled/ht-repro
    rm -f /etc/nginx/sites-enabled/default
    nginx -t && systemctl reload nginx
  EOT

  tags = { Name = "ht-repro" }
}

output "public_ip" { value = aws_instance.ht_repro.public_ip }
output "ssh"       { value = "ssh -i <key.pem> ubuntu@${aws_instance.ht_repro.public_ip}" }
