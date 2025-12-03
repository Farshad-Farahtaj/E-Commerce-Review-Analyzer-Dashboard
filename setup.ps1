# Run This First - Setup and Test Script
# This script helps you get started with the E-Commerce Review Analyzer

Write-Host "=" -NoNewline; Write-Host ("="*79)
Write-Host "E-COMMERCE REVIEW ANALYZER - SETUP SCRIPT"
Write-Host "=" -NoNewline; Write-Host ("="*79)
Write-Host ""

# Function to check if command exists
function Test-Command {
    param($Command)
    try {
        if (Get-Command $Command -ErrorAction Stop) {
            return $true
        }
    } catch {
        return $false
    }
}

# Check Python installation
Write-Host "Step 1: Checking Python installation..." -ForegroundColor Cyan
if (Test-Command python) {
    $pythonVersion = python --version
    Write-Host "  ✓ $pythonVersion found" -ForegroundColor Green
} else {
    Write-Host "  ✗ Python not found! Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Check pip
Write-Host ""
Write-Host "Step 2: Checking pip..." -ForegroundColor Cyan
if (Test-Command pip) {
    Write-Host "  ✓ pip is installed" -ForegroundColor Green
} else {
    Write-Host "  ✗ pip not found!" -ForegroundColor Red
    exit 1
}

# Ask user what they want to do
Write-Host ""
Write-Host "=" -NoNewline; Write-Host ("="*79)
Write-Host "WHAT WOULD YOU LIKE TO DO?"
Write-Host "=" -NoNewline; Write-Host ("="*79)
Write-Host ""
Write-Host "[1] Install dependencies (pip install -r requirements.txt)"
Write-Host "[2] Run the Streamlit dashboard (streamlit run app.py)"
Write-Host "[3] Run accuracy test (python test_accuracy.py)"
Write-Host "[4] Docker build (docker build -t review-analyzer .)"
Write-Host "[5] Docker run (docker run -p 8501:8501 review-analyzer)"
Write-Host "[6] Complete setup (install + run dashboard)"
Write-Host "[0] Exit"
Write-Host ""

$choice = Read-Host "Enter your choice (0-6)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Installing dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
        Write-Host ""
        Write-Host "✓ Installation complete!" -ForegroundColor Green
    }
    "2" {
        Write-Host ""
        Write-Host "Starting Streamlit dashboard..." -ForegroundColor Yellow
        Write-Host "  → Dashboard will open at http://localhost:8501" -ForegroundColor Cyan
        Write-Host "  → Press Ctrl+C to stop the server" -ForegroundColor Cyan
        Write-Host ""
        streamlit run app.py
    }
    "3" {
        Write-Host ""
        Write-Host "Running accuracy assessment..." -ForegroundColor Yellow
        python test_accuracy.py
    }
    "4" {
        if (Test-Command docker) {
            Write-Host ""
            Write-Host "Building Docker image..." -ForegroundColor Yellow
            docker build -t review-analyzer .
            Write-Host ""
            Write-Host "✓ Docker image built successfully!" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "✗ Docker not found! Please install Docker Desktop." -ForegroundColor Red
        }
    }
    "5" {
        if (Test-Command docker) {
            Write-Host ""
            Write-Host "Starting Docker container..." -ForegroundColor Yellow
            Write-Host "  → Dashboard will be available at http://localhost:8501" -ForegroundColor Cyan
            Write-Host "  → Press Ctrl+C to stop the container" -ForegroundColor Cyan
            Write-Host ""
            docker run -p 8501:8501 review-analyzer
        } else {
            Write-Host ""
            Write-Host "✗ Docker not found! Please install Docker Desktop." -ForegroundColor Red
        }
    }
    "6" {
        Write-Host ""
        Write-Host "=" -NoNewline; Write-Host ("="*79)
        Write-Host "COMPLETE SETUP"
        Write-Host "=" -NoNewline; Write-Host ("="*79)
        
        Write-Host ""
        Write-Host "Step 1: Installing dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
        
        Write-Host ""
        Write-Host "✓ Installation complete!" -ForegroundColor Green
        
        Write-Host ""
        Write-Host "Step 2: Starting dashboard..." -ForegroundColor Yellow
        Write-Host "  → Dashboard will open at http://localhost:8501" -ForegroundColor Cyan
        Write-Host "  → Press Ctrl+C to stop the server" -ForegroundColor Cyan
        Write-Host ""
        
        Start-Sleep -Seconds 2
        streamlit run app.py
    }
    "0" {
        Write-Host ""
        Write-Host "Goodbye!" -ForegroundColor Cyan
        exit 0
    }
    default {
        Write-Host ""
        Write-Host "Invalid choice!" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "=" -NoNewline; Write-Host ("="*79)
Write-Host "For more information, check:"
Write-Host "  - README.md for complete documentation"
Write-Host "  - QUICKSTART.md for quick start guide"
Write-Host "  - PROJECT_DELIVERABLES.md for project details"
Write-Host "=" -NoNewline; Write-Host ("="*79)
