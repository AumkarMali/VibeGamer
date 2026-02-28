# Pie – AI Agent (Claude + Chess Bot)

Unified GUI: type a task → screenshot is sent to **Claude** → Claude either starts the **chess bot** (if it sees a board) or comments on the screen in the log.

- **Chess agent:** YOLO piece detection + Stockfish; auto-plays on sites like Lichess/Chess.com.
- **Claude:** Task routing and screen comments (vision). No Llama/Groq.

## Setup

1. **Clone and install**
   ```bash
   git clone https://github.com/AumkarMali/pie.git
   cd pie
   pip install -r requirements.txt
   ```

2. **Stockfish (required for chess)**  
   Download a Windows build from [Stockfish](https://stockfishchess.org/download/) and place `stockfish.exe` in the project root (or use the [latest release](https://github.com/official-stockfish/Stockfish/releases)).

3. **Chess YOLO model (required for chess)**  
   Place the chess piece detection model as `chess_model.pt` in the project root.  
   Example: [NAKSTStudio/yolov8m-chess-piece-detection](https://huggingface.co/NAKSTStudio/yolov8m-chess-piece-detection) – download `best.pt` and rename to `chess_model.pt`.

4. **Claude API key**  
   In the app: **Settings → Claude API Key**. Get a key from [Anthropic](https://console.anthropic.com/).

## Run

```bash
python gui.py
```

Enter a task (e.g. *play chess* or *what’s on my screen?*) and click **Start**. With a Claude key set, the app takes a screenshot and asks Claude; for chess tasks it only starts the bot if Claude sees a board.
