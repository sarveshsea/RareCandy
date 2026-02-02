#!/bin/bash

# Clean Setup Script for Rare Candy
# Handles Docker installation and .env configuration interactively.

echo "💎 Rare Candy Setup Wizard 💎"
echo "------------------------------"

# 1. Environment Configuration
echo "📝 Let's configure your environment..."

if [ -f .env ]; then
    read -p "Existing .env found. Overwrite? (y/N): " confirm
    if [[ "$confirm" != "y" ]]; then
        echo "Using existing .env."
    else
        echo "Creating new .env..."
        rm .env
    fi
fi

if [ ! -f .env ]; then
    read -p "Enter Coinbase API KEY: " api_key
    read -p "Enter Coinbase API SECRET: " api_secret
    
    echo ""
    echo "Select Trading Mode:"
    echo "1) 💸 PAPER TRADING (Live Data, Fake Money) - Recommended for testing"
    echo "2) 🏖️ SANDBOX (Fake Data, Fake Money) - For API connection testing"
    echo "3) 🚀 LIVE (Real Money) - DANGER"
    read -p "Choice (1/2/3): " mode_choice
    
    PAPER_MODE="False"
    SANDBOX_MODE="False"
    
    case $mode_choice in
        1)
            PAPER_MODE="True"
            echo "Selected: PAPER TRADING"
            ;;
        2)
            SANDBOX_MODE="True"
            echo "Selected: SANDBOX"
            ;;
        3)
            echo "Selected: LIVE TRADING"
            ;;
        *)
            echo "Invalid choice. Defaulting to PAPER TRADING."
            PAPER_MODE="True"
            ;;
    esac

    # Create .env
    cat <<EOF > .env
COINBASE_API_KEY="$api_key"
COINBASE_API_SECRET="$api_secret"
SANDBOX_MODE=$SANDBOX_MODE
PAPER_MODE=$PAPER_MODE
EOF
    echo "✅ .env created successfully."
fi

# 2. Docker Setup
echo "------------------------------"
echo "🐳 Checking Docker..."

if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing..."
    # Standard Ubuntu Docker Install
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=\"$(dpkg --print-architecture)\" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker $USER
    echo "✅ Docker installed."
else
    echo "✅ Docker is already installed."
fi

# 3. Launch
echo "------------------------------"
read -p "🚀 Ready to launch? (y/N): " launch
if [[ "$launch" == "y" ]]; then
    # Ensure dashboard dir exists
    mkdir -p dashboard
    
    echo "Building and Starting..."
    if command -v docker-compose &> /dev/null; then
        sudo docker-compose up -d --build
    else
        sudo docker compose up -d --build
    fi
    
    echo "💎 Rare Candy is running!"
    echo "📊 View logs: docker compose logs -f"
else
    echo "Setup complete. Run 'docker compose up -d' when ready."
fi
