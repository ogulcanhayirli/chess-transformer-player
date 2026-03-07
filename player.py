import chess
import random
import torch
from typing import Optional
from chess_tournament import Player


# ============================================================
# OPENING BOOK
# ============================================================
OPENING_BOOK = {
    # --- WHITE OPENINGS ---
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": "e2e4",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": "g1f3",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": "f1c4",
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4": "c2c3",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4": "d2d3",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": "g1f3",
    "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3": "d2d4",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": "d2d4",
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": "d2d4",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": "e4d5",
    "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2": "e4e5",
    # --- BLACK OPENINGS ---
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": "e7e5",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": "e7e5",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1": "d7d5",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1": "d7d5",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": "b8c6",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3": "g8f6",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3": "a7a6",
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2": "e7e6",
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2": "e7e6",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 1 2": "g8f6",
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1": "e7e5",
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1": "e7e5",
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1": "d7d5",
}

PIECE_VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}


def score_move_tactically(board, move):
    bonus = 0.0
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            captured_val = PIECE_VALUES.get(captured.symbol().lower(), 0)
            moving = board.piece_at(move.from_square)
            moving_val = PIECE_VALUES.get(moving.symbol().lower(), 0) if moving else 3
            bonus += captured_val * 3.0 + max(0, captured_val - moving_val)
        else:
            bonus += 3.0
    if move.promotion:
        bonus += 8.0
        if move.promotion == chess.QUEEN:
            bonus += 2.0
    board.push(move)
    if board.is_checkmate():
        bonus += 100.0
    elif board.is_check():
        bonus += 2.0
    if board.is_stalemate():
        bonus -= 50.0
    board.pop()
    return bonus


class TransformerPlayer(Player):

    def __init__(
        self,
        name: str = "ChessGPT",
        model_name: str = "ogulcanhayirli/chess-gpt2-medium",
        device: str = None,
    ):
        super().__init__(name)

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.has_lmfe = False
        try:
            import transformers.tokenization_utils as _tu
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTTB
            _tu.PreTrainedTokenizerBase = _PTTB
            from lmformatenforcer import RegexParser
            from lmformatenforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn,
            )
            self.RegexParser = RegexParser
            self.build_prefix_fn = build_transformers_prefix_allowed_tokens_fn
            self.has_lmfe = True
        except ImportError:
            pass

        self._prompt_cache = {}

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        legal_uci = [m.uci() for m in legal_moves]

        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move.uci()
            board.pop()

        book_move = OPENING_BOOK.get(fen)
        if book_move and book_move in legal_uci:
            return book_move
        fen_parts = fen.split()
        if len(fen_parts) >= 4:
            short_fen = " ".join(fen_parts[:4])
            for book_fen, book_mv in OPENING_BOOK.items():
                book_parts = book_fen.split()
                if len(book_parts) >= 4 and " ".join(book_parts[:4]) == short_fen:
                    if book_mv in legal_uci:
                        return book_mv

        try:
            move = self._score_legal_moves(fen, legal_moves, legal_uci)
            if move and move in legal_uci:
                return move
        except Exception:
            pass

        try:
            move = self._constrained_decode(fen, legal_uci)
            if move and move in legal_uci:
                return move
        except Exception:
            pass

        return random.choice(legal_uci)

    def _score_legal_moves(self, fen, legal_moves, legal_uci):
        prompt = f"FEN: {fen} MOVE: "
        if prompt not in self._prompt_cache:
            self._prompt_cache[prompt] = self.tokenizer.encode(prompt)
        prompt_token_ids = self._prompt_cache[prompt]
        prompt_tensor = torch.tensor([prompt_token_ids], device=self.device)
        prompt_len = len(prompt_token_ids)

        with torch.no_grad():
            prompt_output = self.model(prompt_tensor)
            base_logprobs = torch.log_softmax(prompt_output.logits[0, -1], dim=-1)

        best_score = float("-inf")
        best_move = None

        for move_obj, move_str in zip(legal_moves, legal_uci):
            move_ids = self.tokenizer.encode(move_str, add_special_tokens=False)
            score = base_logprobs[move_ids[0]].item()

            if len(move_ids) > 1:
                full_ids = prompt_token_ids + move_ids
                full_tensor = torch.tensor([full_ids], device=self.device)
                with torch.no_grad():
                    output = self.model(full_tensor)
                    logits = output.logits
                for i in range(1, len(move_ids)):
                    pos = prompt_len + i - 1
                    token_logprob = torch.log_softmax(logits[0, pos], dim=-1)[move_ids[i]].item()
                    score += token_logprob

            board = chess.Board(fen)
            tactical_bonus = score_move_tactically(board, move_obj)
            score += tactical_bonus

            if score > best_score:
                best_score = score
                best_move = move_str

        return best_move

    def _constrained_decode(self, fen, legal_uci):
        if not self.has_lmfe:
            return None
        regex_pattern = "(" + "|".join(legal_uci) + ")"
        parser = self.RegexParser(regex_pattern)
        prefix_fn = self.build_prefix_fn(self.tokenizer, parser)
        prompt = f"FEN: {fen} MOVE: "
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_fn,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        return generated.split()[0] if generated else None
