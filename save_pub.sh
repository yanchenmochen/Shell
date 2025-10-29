#!/bin/bash

# 定义公钥
pub_key="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDUZN3Oh46GlQJlG8FGxYWhl9Xvj3Y0gJ2twSpIUA9ukpXySWpVjQ8am3NZjt1lKL5qVFcRobn8hpPwwZ5coFSN8qon228f85eIWCRSMRvqFpoHfLzC5qHG6hwdq0LXKLfj68q5xNKnSZ3MnB7wA4nTBz1bA5vcq//be3nrGzW5DMl8miwmAvJ0P4xasPPB2iePe6Y2DEHtSgTD3yMGTefq1IzaeZaVEGsrSI8J57vzhqFjOpAnwcPFGwXq/RAESchUX/WHJ498bRijDLCrvYPNQlIzwjx8C74Tj6w/cp8QO2sSRVtuKRf3cuHyB7B69+mUYzrgGHqi7JBGuGSNlMCZ zj@DESKTOP-L6VJN12"

# 创建目录 ~/.ssh 如果不存在
mkdir -p ~/.ssh

# 设置正确的权限
chmod 700 ~/.ssh

# 追加公钥到 authorized_keys 文件
echo "${pub_key}" >> ~/.ssh/authorized_keys

# 设置 authorized_keys 文件的权限
chmod 600 ~/.ssh/authorized_keys