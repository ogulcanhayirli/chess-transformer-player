[README.md](https://github.com/user-attachments/files/25822584/README.md)
# Chess Transformer Player

A transformer-based chess player for the INFOMTALC 2026 Midterm Assignment. Uses a fine-tuned GPT-2 Medium (355M parameters) with inference-time search to play chess at a competitive level.

## Model

**HuggingFace:** [ogulcanhayirli/chess-gpt2-medium](https://huggingface.co/ogulcanhayirli/chess-gpt2-medium)

**Base model:** GPT-2 Medium (355M parameters)

**Training data:** 1.5M positions from [Lichess/chess-position-evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations), filtered for Stockfish depth >= 20

**Format:** `FEN: <fen_string> MOVE: <uci_move>`

**Training config:** 3 epochs, batch size 32, learning rate 3e-5, max sequence length 128

## How It Works

The player combines the fine-tuned transformer with several inference-time techniques:

1. **Opening Book:** Hardcoded opening theory for the first moves, where the model has less training coverage.

2. **Log-Likelihood Move Scoring:** Instead of generating and hoping for a legal move, the player computes `P(move | FEN)` for every legal move and picks the highest-scoring one. This guarantees zero fallbacks.

3. **1-Ply Lookahead Search:** For the top 10 candidate moves, the player simulates each on the board and scores the opponent's best response. The move that maximizes our advantage while minimizing the opponent's best option is selected. This turns the transformer into an evaluation function within a minimax-style framework.

4. **Tactical Safety Net:** Penalizes moves that leave pieces undefended and attacked, preventing common neural network blunders like hanging a queen.

5. **Material-Aware Tiebreaking:** When model scores are similar, prefers moves that capture material.

## Repository Structure

```
player.py            # TransformerPlayer class (main submission)
requirements.txt     # Python dependencies
README.md            # This file
```

## Usage

```python
from chess_tournament import Game, RandomPlayer
from player import TransformerPlayer

player = TransformerPlayer("MyPlayer")
opponent = RandomPlayer("Random")

game = Game(player, opponent, max_half_moves=200)
result = game.play()
print(result)
```

## Requirements

```
torch>=2.0.0
transformers>=4.38.0
accelerate>=0.26.0
python-chess>=1.999
```

## Setup

```bash
git clone https://github.com/bylinina/chess_exam.git
cd chess_exam
pip install -e .
pip install -r requirements.txt
```

## Author

Ogulcan Hayirli, Utrecht University, MSc Applied Data Science
