# AWS Setup Guide for Multi-GPU Testing

This guide will help you set up an AWS instance for running multi-GPU tests.

## 1. Launch an AWS Instance

1. Go to the AWS Console and navigate to EC2
2. Click "Launch Instance"
3. Choose an instance type:
   - For 2 GPUs: `g4dn.xlarge` (cheaper option)
   - For 4 GPUs: `p3.8xlarge` (more powerful)
   - For 8 GPUs: `p3.16xlarge` (most powerful)
4. Choose Ubuntu 20.04 LTS as the AMI
5. Configure instance details:
   - Number of instances: 1
   - Network: default VPC
   - Subnet: any availability zone
6. Add storage:
   - Root volume: 30 GB (default)
7. Add tags (optional)
8. Configure security group:
   - Allow SSH (port 22) from your IP
9. Review and launch
10. Create or select an existing key pair
11. Download the key pair file (.pem)

## 2. Connect to the Instance

```bash
# Make the key file secure
chmod 400 your-key-pair.pem

# Connect to the instance
ssh -i your-key-pair.pem ubuntu@your-instance-public-dns
```

## 3. Set Up the Environment

1. Clone the repository:
```bash
git clone https://github.com/your-repo/transformer-lens.git
cd transformer-lens
```

2. Make the setup script executable:
```bash
chmod +x scripts/setup_aws_gpu.sh
```

3. Run the setup script:
```bash
./scripts/setup_aws_gpu.sh
```

## 4. Run the Tests

1. Make the test runner executable:
```bash
chmod +x scripts/run_gpu_tests.sh
```

2. Run the tests:
```bash
./scripts/run_gpu_tests.sh
```

3. Check the results:
```bash
cat test_results/*.log
```

## 5. Clean Up

When you're done, make sure to:
1. Stop the EC2 instance to avoid unnecessary charges
2. Consider terminating the instance if you won't need it again

## Troubleshooting

1. If you get a "No space left on device" error:
```bash
df -h  # Check disk usage
sudo resize2fs /dev/nvme0n1p1  # Resize the root partition
```

2. If CUDA installation fails:
```bash
sudo apt-get install -y nvidia-cuda-toolkit
```

3. If PyTorch can't find CUDA:
```bash
python3 -c "import torch; print(torch.version.cuda)"
# Should match the CUDA version installed
```

## Cost Estimation

- g4dn.xlarge: ~$0.526/hour
- p3.8xlarge: ~$3.06/hour
- p3.16xlarge: ~$12.24/hour

Remember to stop the instance when not in use to avoid unnecessary charges. 