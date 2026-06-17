# Activity Tracker

> OCR-powered screenshot activity logger. Upload screenshots, auto-detect apps, extract text, build a timeline.

[![GitHub stars](https://img.shields.io/github/stars/jellydn/activity-tracker)](https://github.com/jellydn/activity-tracker/stargazers)
[![GitHub license](https://img.shields.io/github/license/jellydn/activity-tracker)](https://github.com/jellydn/activity-tracker/blob/main/LICENSE)

## Features

- **OCR Text Extraction** - RapidOCR (ONNX Runtime), no external dependencies
- **Auto App Detection** - Recognizes 30+ apps (VS Code, Chrome, Slack, GitHub, Figma, etc.)
- **Smart Tagging** - Auto-tags activities (Debugging, Git, DevOps, Testing, API, etc.)
- **Timeline View** - Visual history with thumbnails and metadata
- **Export** - Download all data as JSON
- **Zero Config** - Single Python file, SQLite backend, runs anywhere

## Tech Stack

- **FastAPI** + **Uvicorn** - Async Python web framework
- **RapidOCR** - Pure Python OCR engine (ONNX Runtime)
- **SQLite** - Zero-config database
- **Pillow** - Image processing
- **uv** - Modern Python package manager

## Quick Start

```bash
# Clone
git clone https://github.com/jellydn/activity-tracker.git
cd activity-tracker

# Install dependencies
uv sync

# Run
uv run uvicorn main:app --host 0.0.0.0 --port 8090
```

Open `http://localhost:8090` in your browser.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend (single-page app) |
| `/api/upload` | POST | Upload screenshot (multipart) |
| `/api/records` | GET | List all records |
| `/api/stats` | GET | Dashboard statistics |
| `/api/export` | GET | Export all data as JSON |
| `/api/records` | DELETE | Clear all records |

## Usage

1. **Upload** - Drag & drop, click to browse, or paste from clipboard (Ctrl+V)
2. **OCR** - Automatic text extraction with confidence score
3. **Detect** - App, URL, content type, and activity tags auto-detected
4. **Timeline** - Browse history with thumbnails and metadata
5. **Export** - Download all data as JSON for backup

## Project Structure

```
activity-tracker/
├── main.py           # FastAPI backend + inline frontend (~250 lines)
├── pyproject.toml    # Dependencies (uv)
└── data/             # SQLite DB + thumbnails (auto-created)
```

## Development

```bash
# Lint
ruff check main.py

# Type check
ty check main.py
```

## License

MIT
