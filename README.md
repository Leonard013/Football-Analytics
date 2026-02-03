# Football Analytics

Real-time football match video analytics using YOLOv8. Detects players and ball, identifies teams by jersey color via K-means clustering, and tracks per-team stats: ball possession, tackles, passages, and out-of-bounds events. Streams annotated video and live stats to a web browser via Flask + SocketIO.

## Setup

Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

**Linux (NVIDIA GPU):**
```bash
conda env create -f environment.yml
```

**macOS (Apple Silicon):**
```bash
conda env create -f environment-macos.yml
```

Then:
```bash
conda activate football-analytics
```

## Usage

Place a video file at `video/match.mp4`, then:

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

The app auto-detects the best available device (CUDA → MPS → CPU).

## License

MIT
