#!/bin/bash

# Define Colors for "Cool" UI
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
RESET='\033[0m'
BOLD='\033[1m'

# Helper for nice headers
function print_header() {
    echo -e "\n${BLUE}${BOLD}┌───────────────────────────────────────────────────┐${RESET}"
    echo -e "${BLUE}${BOLD}│ $1${RESET}"
    echo -e "${BLUE}${BOLD}└───────────────────────────────────────────────────┘${RESET}"
}

function print_step() {
    echo -e "${CYAN}➜${RESET} $1"
}

function print_success() {
    echo -e "${GREEN}✔${RESET} $1"
}

clear
echo -e "${CYAN}${BOLD}"
cat << "EOF"
                                  .
                                . : .
                              . : : : .
                            . : : : : : .
                          . : : : : : : : .
                        .' : : : : : : : : '.
                        ; : : : : : : : : : ;
                        | : : : RARE  : : : |
                        | : : : CANDY : : : |
                        ; : : : : : : : : : ;
                        '. : : : : : : : : .'
                          ' . : : : : : . '
                            ' . : : : . '
                              ' . : . '
                                ' . '
                                
             (o)   (o)   (o)   (o)   (o)   (o)
             
       💎  W E L C O M E   T O   L E V E L   1 0 0  💎
EOF
echo -e "${RESET}"

# 1. Environment Configuration
print_header "Step 1: Configuration"

if [ -f .env ]; then
    print_step "Existing .env found."
    read -p "   Overwrite? (y/N): " confirm
    if [[ "$confirm" != "y" ]]; then
        print_success "Using existing .env."
    else
        rm .env
        print_step "Creating new configuration..."
    fi
fi

if [ ! -f .env ]; then
    echo -e "${YELLOW}Please enter your COINBASE API credentials:${RESET}"
    read -p "   API KEY: " api_key
    read -p "   API SECRET: " api_secret
    
    echo ""
    print_step "Select Trading Mode:"
    echo -e "   ${GREEN}1) 💸 PAPER TRADING${RESET} (Recommended)"
    echo -e "   ${YELLOW}2) 🏖️ SANDBOX${RESET} (Dev)"
    echo -e "   ${RED}3) 🚀 LIVE TRADING${RESET} (Real Money)"
    read -p "   Choice (1-3): " mode_choice
    
    PAPER_MODE="False"
    SANDBOX_MODE="False"
    
    case $mode_choice in
        1)
            PAPER_MODE="True"
            print_success "Mode set to PAPER."
            ;;
        2)
            SANDBOX_MODE="True"
            print_success "Mode set to SANDBOX."
            ;;
        3)
            echo -e "${RED}⚠️  Mode set to LIVE.${RESET}"
            ;;
        *)
            PAPER_MODE="True"
            print_success "Defaulting to PAPER."
            ;;
    esac

    # Create .env
    cat <<EOF > .env
COINBASE_API_KEY="$api_key"
COINBASE_API_SECRET="$api_secret"
SANDBOX_MODE=$SANDBOX_MODE
PAPER_MODE=$PAPER_MODE
EOF
    print_success ".env configurations saved."
fi

# 2. Docker Setup
print_header "Step 2: System Setup"

if ! command -v docker &> /dev/null; then
    print_step "Docker not found. Installing..."
    
    # Supress apt output for cleaner look
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=\"$(dpkg --print-architecture)\" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker $USER
    print_success "Docker installed successfully."
else
    print_success "Docker is already installed."
fi

# 3. Launch
print_header "Step 3: Launch"

print_step "Ready to build and start containers."
read -p "   Start now? (y/N): " launch
if [[ "$launch" == "y" ]]; then
    print_step "Building containers (this may take a moment)..."
    
    mkdir -p dashboard
    
    if command -v docker-compose &> /dev/null; then
        sudo docker-compose up -d --build
    else
        sudo docker compose up -d --build
    fi
    
    print_header "🚀 DEPLOYMENT COMPLETE"
    echo -e "   ${GREEN}Bot is running in background.${RESET}"
    echo -e "   ${BLUE}Logs:${RESET}      docker compose logs -f"
    echo -e "   ${BLUE}Dashboard:${RESET} http://$(curl -s ifconfig.me):8000/status.json"
else
    print_success "Setup complete. Run 'docker compose up -d' when ready."
fi
