# Image Filter Application

A Flask-based web application that allows users to upload images and apply various filters to them.

## Features

- Upload images (PNG, JPG, JPEG)
- Apply different filters:
  - Grayscale
  - Blur
  - Edge Detection
  - Sharpen
  - Sepia
- View original and filtered images side by side
- Responsive web interface

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install uv (if not already installed):
```bash
# On Windows (PowerShell):
iwr https://astral.sh/uv/install.ps1 -useb | iex

# On Unix-like systems:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install dependencies using uv:
```bash
uv sync
```
This will automatically create a virtual environment (`.venv`) and install all required dependencies.

## Usage

1. Activate the virtual environment:
```bash
# On Windows:
.\.venv\Scripts\activate

# On Unix-like systems:
source .venv/bin/activate
```

2. Start the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

4. Upload an image and select a filter to apply.

## Project Structure

```
.
├── app.py              # Main Flask application
├── pyproject.toml      # Project configuration and dependencies
├── requirements.txt    # Legacy requirements file
├── static/            # Static files (uploaded images)
│   └── uploads/       
└── templates/         # HTML templates
    └── index.html     
```

## Package Management

This project uses `uv` for dependency management. Common commands:

- Install a new package: `uv pip install package_name`
- Update a package: `uv pip install --upgrade package_name`
- Remove a package: `uv pip uninstall package_name`
- List installed packages: `uv pip list`
- Sync dependencies: `uv sync`

## Supported Image Formats

- PNG
- JPG/JPEG

## File Size Limit

Maximum file size: 16MB