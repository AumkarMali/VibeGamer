"""
Chess Engine Module - Headless chess agent (no UI).

Provides ChessEngine: YOLO piece detection + Stockfish analysis + move execution.
The main GUI drives this engine and handles all display/logging.
"""
import os
import time
import numpy as np
import pyautogui
import chess
from PIL import Image, ImageDraw
from ultralytics import YOLO
from stockfish import Stockfish

# ── paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chess_model.pt")
STOCKFISH_PATH = os.path.join(BASE_DIR, "stockfish.exe")

# ── game-over keywords (lowercase) ──
GAME_OVER_KEYWORDS = [
    "won", "wins", "lost", "lose", "aborted", "draw", "drawn",
    "stalemate", "checkmate", "resigned", "timeout", "victory",
    "defeat", "game over", "1-0", "0-1", "1/2", "rematch",
]

# ── YOLO class -> python-chess piece mapping ──
YOLO_TO_PIECE = {
    "white_king":   chess.Piece(chess.KING,   chess.WHITE),
    "white_queen":  chess.Piece(chess.QUEEN,  chess.WHITE),
    "white_rook":   chess.Piece(chess.ROOK,   chess.WHITE),
    "white_bishop": chess.Piece(chess.BISHOP, chess.WHITE),
    "white_knight": chess.Piece(chess.KNIGHT, chess.WHITE),
    "white_pawn":   chess.Piece(chess.PAWN,   chess.WHITE),
    "black_king":   chess.Piece(chess.KING,   chess.BLACK),
    "black_queen":  chess.Piece(chess.QUEEN,  chess.BLACK),
    "black_rook":   chess.Piece(chess.ROOK,   chess.BLACK),
    "black_bishop": chess.Piece(chess.BISHOP, chess.BLACK),
    "black_knight": chess.Piece(chess.KNIGHT, chess.BLACK),
    "black_pawn":   chess.Piece(chess.PAWN,   chess.BLACK),
}

# Keywords that trigger chess mode
CHESS_KEYWORDS = [
    "chess", "play chess", "chess game", "chess agent",
    "chess match", "stockfish", "lichess", "chess.com",
]


# ======================================================================
#  Board-state helpers
# ======================================================================

def detections_to_board(pieces, board_box):
    """Map YOLO-detected pieces onto an 8x8 chess.Board."""
    if board_box:
        bx1, by1 = board_box["x1"], board_box["y1"]
        bx2, by2 = board_box["x2"], board_box["y2"]
    else:
        xs = [p["cx"] for p in pieces]
        ys = [p["cy"] for p in pieces]
        pad_x = (max(xs) - min(xs)) * 0.06
        pad_y = (max(ys) - min(ys)) * 0.06
        bx1, by1 = min(xs) - pad_x, min(ys) - pad_y
        bx2, by2 = max(xs) + pad_x, max(ys) + pad_y

    sq_w = (bx2 - bx1) / 8
    sq_h = (by2 - by1) / 8

    white_y, white_n = 0, 0
    black_y, black_n = 0, 0
    for p in pieces:
        if p["label"].startswith("white"):
            white_y += p["cy"]; white_n += 1
        elif p["label"].startswith("black"):
            black_y += p["cy"]; black_n += 1

    orientation = ("white"
                   if (white_y / max(white_n, 1)) > (black_y / max(black_n, 1))
                   else "black")

    board = chess.Board(fen=None)
    board.clear()

    for p in pieces:
        label = p["label"]
        if label not in YOLO_TO_PIECE:
            continue
        piece = YOLO_TO_PIECE[label]
        col = max(0, min(7, int((p["cx"] - bx1) / sq_w)))
        row = max(0, min(7, int((p["cy"] - by1) / sq_h)))
        if orientation == "white":
            sq = chess.square(col, 7 - row)
        else:
            sq = chess.square(7 - col, row)
        board.set_piece_at(sq, piece)

    # Infer castling rights
    board.castling_rights = chess.BB_EMPTY
    if board.king(chess.WHITE) == chess.E1:
        if board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
            board.castling_rights |= chess.BB_H1
        if board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
            board.castling_rights |= chess.BB_A1
    if board.king(chess.BLACK) == chess.E8:
        if board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
            board.castling_rights |= chess.BB_H8
        if board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
            board.castling_rights |= chess.BB_A8

    return board, orientation


def board_ascii(board, orientation):
    """Pretty-print the board from the given orientation."""
    lines = []
    if orientation == "white":
        for rank in range(7, -1, -1):
            row = f" {rank+1}  "
            for f in range(8):
                p = board.piece_at(chess.square(f, rank))
                row += f" {p.symbol() if p else '.'}"
            lines.append(row)
        lines.append("     a b c d e f g h")
    else:
        for rank in range(8):
            row = f" {rank+1}  "
            for f in range(7, -1, -1):
                p = board.piece_at(chess.square(f, rank))
                row += f" {p.symbol() if p else '.'}"
            lines.append(row)
        lines.append("     h g f e d c b a")
    return "\n".join(lines)


def square_to_screen(sq_name, board_box, orientation):
    """Convert chess square name (e.g. 'e2') to screen pixel center."""
    bx1, by1 = board_box["x1"], board_box["y1"]
    sq_w = (board_box["x2"] - bx1) / 8
    sq_h = (board_box["y2"] - by1) / 8
    fi = ord(sq_name[0]) - ord('a')
    ri = int(sq_name[1]) - 1
    if orientation == "white":
        px = bx1 + fi * sq_w + sq_w / 2
        py = by1 + (7 - ri) * sq_h + sq_h / 2
    else:
        px = bx1 + (7 - fi) * sq_w + sq_w / 2
        py = by1 + ri * sq_h + sq_h / 2
    return int(px), int(py)


def is_chess_task(text):
    """Return True if the task text looks like a chess request."""
    t = text.lower()
    return any(kw in t for kw in CHESS_KEYWORDS)


# ======================================================================
#  Chess Engine (headless)
# ======================================================================

class ChessEngine:
    """YOLO + Stockfish chess engine with no UI.

    Parameters
    ----------
    log_fn : callable(msg, tag)
        Function the engine calls to emit log messages.
    """

    def __init__(self, log_fn=None):
        self._log = log_fn or (lambda msg, tag="": None)
        self.model = None
        self.engine = None
        self.ready = False

        # per-game state
        self._last_fen = None
        self._move_count = 0
        self._cycle_count = 0
        self._unchanged_count = 0
        self._ocr_reader = None

    def log(self, msg, tag=""):
        self._log(msg, tag)

    # ------------------------------------------------------------------
    #  Loading
    # ------------------------------------------------------------------
    def load_models(self):
        """Load YOLO and Stockfish. Returns True on success."""
        ok = True
        try:
            t0 = time.time()
            self.model = YOLO(MODEL_PATH)
            self.model.predict(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
            self.log(f"  YOLO chess model loaded ({time.time()-t0:.1f}s)", "info")
        except Exception as e:
            self.log(f"  YOLO FAILED: {e}", "error")
            ok = False

        try:
            t0 = time.time()
            if not os.path.isfile(STOCKFISH_PATH):
                raise FileNotFoundError(f"stockfish.exe not found")
            self.engine = Stockfish(
                path=STOCKFISH_PATH, depth=18,
                parameters={"Threads": 2, "Hash": 128})
            self.log(f"  Stockfish loaded ({time.time()-t0:.1f}s)", "info")
        except Exception as e:
            self.log(f"  STOCKFISH FAILED: {e}", "error")
            ok = False

        self.ready = ok
        return ok

    # ------------------------------------------------------------------
    #  State
    # ------------------------------------------------------------------
    def reset(self):
        """Reset per-game state for a new game."""
        self._last_fen = None
        self._move_count = 0
        self._cycle_count = 0
        self._unchanged_count = 0

    @property
    def move_count(self):
        return self._move_count

    @property
    def cycle_count(self):
        return self._cycle_count

    # ------------------------------------------------------------------
    #  Main analysis
    # ------------------------------------------------------------------
    def analyze(self, screenshot, conf, depth, turn, force_move=False):
        """Run one full cycle on a screenshot.

        Returns a dict:
            status   : "move" | "waiting" | "game_over" | "no_board" | "error"
            best_move: e.g. "e2e4" (only when status=="move")
            eval     : e.g. "+0.35" or "M3"
            fen      : full FEN string
            pieces   : list of piece dicts
            board_box: board bounding box dict or None
            orientation: "white" or "black"
            annotated: PIL Image with detection overlay
            top_moves: list of top moves from Stockfish
        """
        self._cycle_count += 1

        res = dict(status="error", best_move=None, eval=None, fen=None,
                   pieces=[], board_box=None, orientation="white",
                   annotated=None, top_moves=[], message="")

        # ── YOLO detection ──
        img_arr = np.array(screenshot.convert("RGB"))
        t0 = time.time()
        results = self.model.predict(img_arr, conf=conf, verbose=False)
        dt = time.time() - t0

        pieces, board_box = [], None
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0])
                label = r.names[cls_id]
                c = float(box.conf[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if label == "board":
                    board_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": c}
                else:
                    pieces.append({"label": label, "confidence": c,
                                   "cx": cx, "cy": cy,
                                   "x1": x1, "y1": y1, "x2": x2, "y2": y2})

        res["pieces"] = pieces
        res["board_box"] = board_box

        if not pieces:
            self.log(f"  No pieces found ({dt:.2f}s)", "dim")
            res["status"] = "no_board"
            return res

        # ── Map to board ──
        board, orientation = detections_to_board(pieces, board_box)
        board.turn = chess.WHITE if turn == "white" else chess.BLACK
        fen = board.fen()
        fen_pieces = fen.split(" ")[0]

        res["fen"] = fen
        res["orientation"] = orientation

        # ── Board unchanged? ──
        if self._last_fen == fen_pieces and not force_move:
            self._unchanged_count += 1
            self.log(f"  Board unchanged, waiting... ({dt:.2f}s)", "dim")

            if self._unchanged_count >= 2:
                self.log("  Checking for game-over text...", "dim")
                kw = self._check_game_over(screenshot, board_box)
                if kw:
                    self.log(f"  GAME OVER detected: \"{kw}\"", "action")
                    res["status"] = "game_over"
                    res["message"] = kw
                    return res

            res["status"] = "waiting"
            res["annotated"] = self._draw(screenshot, pieces, board_box,
                                          orientation=orientation)
            return res

        # ── Board changed ──
        self._unchanged_count = 0
        self.log(f"  {len(pieces)} pieces in {dt:.2f}s", "info")

        counts = {}
        for p in pieces:
            counts[p["label"]] = counts.get(p["label"], 0) + 1
        cstr = ", ".join(f"{n.split('_')[1][0].upper()}{c}"
                         for n, c in sorted(counts.items()) if n != "board")
        self.log(f"  Pieces: {cstr}", "piece")
        self.log(f"  FEN: {fen}", "action")

        for line in board_ascii(board, orientation).split("\n"):
            self.log(f"  {line}", "board")

        res["annotated"] = self._draw(screenshot, pieces, board_box,
                                      orientation=orientation)

        # ── Validate kings ──
        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            missing = []
            if board.king(chess.WHITE) is None:
                missing.append("white king")
            if board.king(chess.BLACK) is None:
                missing.append("black king")
            self._unchanged_count += 1
            self.log(f"  SKIP: Missing {', '.join(missing)} "
                     f"(#{self._unchanged_count})", "error")

            self.log("  Checking for game-over text...", "dim")
            kw = self._check_game_over(screenshot, board_box)
            if kw:
                self.log(f"  *** GAME OVER: \"{kw}\" ***", "action")
                res["status"] = "game_over"
                res["message"] = kw
                return res

            if self._unchanged_count >= 3:
                self.log("  Board obstructed 3x - game likely over", "action")
                res["status"] = "game_over"
                res["message"] = "board obstructed"
                return res

            res["status"] = "error"
            return res

        # ── Stockfish ──
        try:
            t0 = time.time()
            self.engine.set_depth(depth)
            self.engine.set_fen_position(fen)
            best = self.engine.get_best_move()
            top = self.engine.get_top_moves(3)
            ev = self.engine.get_evaluation()
            et = time.time() - t0

            ev_str = (f"{ev['value']/100:+.2f}" if ev["type"] == "cp"
                      else f"M{ev['value']}")
            res["eval"] = ev_str
            res["top_moves"] = top
        except Exception as e:
            self.log(f"  Stockfish error: {e}", "error")
            self._restart_engine()
            res["status"] = "error"
            return res

        if not best:
            self.log("  No legal move (game over?)", "warning")
            res["status"] = "game_over"
            res["message"] = "no legal moves"
            return res

        # ── Success ──
        res["best_move"] = best
        res["status"] = "move"

        fsq, tsq = best[:2], best[2:4]
        promo = f"={best[4:].upper()}" if len(best) > 4 else ""
        self.log(f"  >> BEST: {fsq} -> {tsq}{promo}  "
                 f"eval={ev_str}  ({et:.2f}s)", "move")

        for i, m in enumerate(top[:3], 1):
            mv = m["Move"]
            sc = (f"M{m['Mate']}" if m["Mate"] is not None
                  else f"{m['Centipawn']/100:+.1f}")
            self.log(f"     {i}. {mv[:2]}->{mv[2:4]}  ({sc})", "dim")

        res["annotated"] = self._draw(screenshot, pieces, board_box,
                                      best_move=best, orientation=orientation)
        return res

    # ------------------------------------------------------------------
    #  Move execution
    # ------------------------------------------------------------------
    def execute_move(self, best_move, board_box, orientation, click_delay=0.15):
        """Click to make the move on screen (call between hide/show)."""
        fsq, tsq = best_move[:2], best_move[2:4]
        promo = best_move[4:] if len(best_move) > 4 else None

        fx, fy = square_to_screen(fsq, board_box, orientation)
        tx, ty = square_to_screen(tsq, board_box, orientation)

        pyautogui.click(fx, fy)
        time.sleep(click_delay)
        pyautogui.click(tx, ty)
        time.sleep(click_delay)

        if promo:
            time.sleep(click_delay)
            pyautogui.click(tx, ty)

        self._move_count += 1

    # ------------------------------------------------------------------
    #  Post-move re-scan
    # ------------------------------------------------------------------
    def capture_post_move_fen(self, screenshot, conf):
        """Re-detect the board after our move to track opponent changes."""
        try:
            img_arr = np.array(screenshot.convert("RGB"))
            results = self.model.predict(img_arr, conf=conf, verbose=False)

            pieces, board_box = [], None
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    cls_id = int(box.cls[0])
                    label = r.names[cls_id]
                    c = float(box.conf[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if label == "board":
                        board_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": c}
                    else:
                        pieces.append({"label": label, "confidence": c,
                                       "cx": cx, "cy": cy,
                                       "x1": x1, "y1": y1, "x2": x2, "y2": y2})

            if pieces:
                board, _ = detections_to_board(pieces, board_box)
                self._last_fen = board.fen().split(" ")[0]
                self.log("  Saved post-move board state", "dim")
            else:
                self._last_fen = None
        except Exception:
            self._last_fen = None

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _restart_engine(self):
        try:
            self.engine = Stockfish(
                path=STOCKFISH_PATH, depth=18,
                parameters={"Threads": 2, "Hash": 128})
            self.log("  Engine restarted", "info")
        except Exception:
            self.log("  Engine restart FAILED", "error")

    def _check_game_over(self, screenshot, board_box):
        """OCR the board area for game-over keywords. Returns keyword or None."""
        try:
            if self._ocr_reader is None:
                self.log("  Loading OCR reader (first time)...", "dim")
                import easyocr
                self._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self.log("  OCR reader ready", "dim")

            if board_box:
                pad = 120
                left = max(0, board_box["x1"] - pad)
                top = max(0, board_box["y1"] - pad)
                right = min(screenshot.width, board_box["x2"] + pad)
                bottom = min(screenshot.height, board_box["y2"] + pad)
            else:
                w, h = screenshot.width, screenshot.height
                left, top = int(w * 0.25), int(h * 0.15)
                right, bottom = int(w * 0.75), int(h * 0.85)

            crop = screenshot.crop((left, top, right, bottom))
            crop_arr = np.array(crop.convert("RGB"))
            results = self._ocr_reader.readtext(crop_arr, detail=0)
            text = " ".join(results).lower()
            self.log(f"  OCR text: \"{text[:120]}\"", "dim")

            for kw in GAME_OVER_KEYWORDS:
                if kw in text:
                    return kw

            self.log("  No game-over keyword matched", "dim")
        except Exception as e:
            self.log(f"  OCR error: {e}", "error")
        return None

    def _draw(self, screenshot, pieces, board_box,
              best_move=None, orientation="white"):
        """Return annotated screenshot with detections and move highlights."""
        img = screenshot.copy()
        draw = ImageDraw.Draw(img)

        if board_box:
            draw.rectangle([board_box["x1"], board_box["y1"],
                            board_box["x2"], board_box["y2"]],
                           outline="#FFD700", width=3)

        pc = {"king": "#FF1493", "queen": "#FFD700", "rook": "#FF8800",
              "bishop": "#AA00FF", "knight": "#00AAFF", "pawn": "#AAAAAA"}

        for p in pieces:
            color = "#00FF00"
            for key, c in pc.items():
                if key in p["label"]:
                    color = c
                    break
            if "black" in p["label"]:
                color = "#" + "".join(
                    format(max(0, int(color[i:i+2], 16) - 60), "02x")
                    for i in (1, 3, 5))
            draw.rectangle([p["x1"], p["y1"], p["x2"], p["y2"]],
                           outline=color, width=2)
            lbl = f"{p['label'].split('_')[1]} {p['confidence']:.0%}"
            tb = draw.textbbox((p["x1"], p["y1"] - 14), lbl)
            draw.rectangle(tb, fill=color)
            draw.text((p["x1"], p["y1"] - 14), lbl, fill="black")

        if best_move and board_box:
            bx1, by1 = board_box["x1"], board_box["y1"]
            sw = (board_box["x2"] - bx1) / 8
            sh = (board_box["y2"] - by1) / 8
            for sq_name, sq_color in [(best_move[:2], "#00FF00"),
                                      (best_move[2:4], "#FFFF00")]:
                fi = ord(sq_name[0]) - ord('a')
                ri = int(sq_name[1]) - 1
                if orientation == "white":
                    px, py = bx1 + fi * sw, by1 + (7 - ri) * sh
                else:
                    px, py = bx1 + (7 - fi) * sw, by1 + ri * sh
                draw.rectangle([int(px), int(py),
                                int(px + sw), int(py + sh)],
                               outline=sq_color, width=4)
        return img
