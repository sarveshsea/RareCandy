#!/bin/bash

# Define Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}"
cat << "EOF"
  _____     _ _                _      
 |_   _|   (_) |              | |     
   | | __ _ _| |___  ___  __ _| | ___ 
   | |/ _` | | / __|/ __|/ _` | |/ _ \

   🔒 SECURE MESH ACCESS SETUP
EOF
echo -e "${RESET}"

# 1. Install Tailscale
echo -e "${CYAN}➜ Installing Tailscale...${RESET}"
curl -fsSL https://tailscale.com/install.sh | sh

# 2. Authenticate
echo -e "\n${CYAN}➜ Authenticating...${RESET}"
echo -e "${YELLOW}Please authenticate via the link below:${RESET}"
sudo tailscale up

# 3. Get IP
TS_IP=$(tailscale ip -4)

echo -e "\n${GREEN}✔ Tailscale Setup Complete!${RESET}"
echo -e "---------------------------------------------------"
echo -e "Your Droplet's Secure IP: ${BOLD}${TS_IP}${RESET}"
echo -e "Dashboard URL: ${BOLD}http://${TS_IP}:8000/status.json${RESET}"
echo -e "---------------------------------------------------"
