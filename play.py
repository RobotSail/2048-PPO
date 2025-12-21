"""
Interactive 2048 game client using Pygame Zero.
Run with: pgzrun play.py
"""
import pgzrun
from game import Game2048, Direction

# Window settings
WIDTH = 500
HEIGHT = 550
TITLE = "2048 Game"

# Colors
COLORS = {
    0: (205, 193, 180),
    1: (238, 228, 218),    # 2
    2: (237, 224, 200),    # 4
    3: (242, 177, 121),    # 8
    4: (245, 149, 99),     # 16
    5: (246, 124, 95),     # 32
    6: (246, 94, 59),      # 64
    7: (237, 207, 114),    # 128
    8: (237, 204, 97),     # 256
    9: (237, 200, 80),     # 512
    10: (237, 197, 63),    # 1024
    11: (237, 194, 46),    # 2048
}

# Game state
game = Game2048()
game.reset()
game_over = False

def draw():
    """Draw the game grid and tiles."""
    screen.clear()
    screen.fill((187, 173, 160))

    # Draw title and score
    screen.draw.text(
        "2048",
        topleft=(20, 20),
        fontsize=60,
        color="white",
        fontname="arial"
    )

    score_text = f"Score: {game.get_score()}"
    screen.draw.text(
        score_text,
        topright=(WIDTH - 20, 30),
        fontsize=40,
        color="white",
        fontname="arial"
    )

    # Draw grid
    tile_size = 100
    padding = 10
    offset_x = 50
    offset_y = 120

    for row in range(4):
        for col in range(4):
            x = offset_x + col * (tile_size + padding)
            y = offset_y + row * (tile_size + padding)

            power = game.grid[row][col]
            color = COLORS.get(power, COLORS[11])

            # Draw tile background
            screen.draw.filled_rect(
                Rect(x, y, tile_size, tile_size),
                color
            )

            # Draw number if not empty
            if power > 0:
                value = 2 ** power
                text_color = "white" if power > 2 else (119, 110, 101)
                fontsize = 60 if value < 1000 else 45

                screen.draw.text(
                    str(value),
                    center=(x + tile_size // 2, y + tile_size // 2),
                    fontsize=fontsize,
                    color=text_color,
                    fontname="arial"
                )

    # Draw game over message
    if game_over:
        screen.draw.text(
            "GAME OVER!",
            center=(WIDTH // 2, HEIGHT // 2),
            fontsize=50,
            color="red",
            fontname="arial"
        )
        screen.draw.text(
            "Press R to restart",
            center=(WIDTH // 2, HEIGHT // 2 + 60),
            fontsize=30,
            color="white",
            fontname="arial"
        )

def on_key_down(key):
    """Handle keyboard input."""
    global game_over

    if key == keys.R:
        # Restart game
        game.reset()
        game_over = False
        return

    if game_over:
        return

    # Map keys to directions
    direction = None
    if key == keys.UP or key == keys.W:
        direction = Direction.UP
    elif key == keys.DOWN or key == keys.S:
        direction = Direction.DOWN
    elif key == keys.LEFT or key == keys.A:
        direction = Direction.LEFT
    elif key == keys.RIGHT or key == keys.D:
        direction = Direction.RIGHT

    if direction:
        _, _, done, _ = game.step(direction)
        if done:
            game_over = True

# Start the game
pgzrun.go()
