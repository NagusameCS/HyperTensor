$env:PATH += ";C:\Program Files\Amazon\AWSCLIV2"

$bdm = @'
[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":80,"VolumeType":"gp3"}}]
'@
$mktopt = @'
{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}
'@

$bdm | Set-Content "$env:TEMP\bdm.json" -Encoding UTF8
$mktopt | Set-Content "$env:TEMP\mktopt.json" -Encoding UTF8

aws ec2 run-instances `
  --image-id ami-039069d7b00819a90 `
  --instance-type g5.xlarge `
  --key-name hypertensor-key `
  --instance-market-options "file://$env:TEMP\mktopt.json" `
  --block-device-mappings "file://$env:TEMP\bdm.json" `
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=hypertensor-dev}]" `
  --region us-east-1 `
  --query "Instances[0].{ID:InstanceId,State:State.Name,Type:InstanceType}"
