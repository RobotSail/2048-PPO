"""
CLI 2048 game client for terminal play.
Run with: python play_cli.py
"""
import sys
import tty
import termios
from game import Game2048, Direction

def get_key():
    """Get a single keypress from the terminal."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        # Handle arrow keys (they send 3 characters: ESC [ A/B/C/D)
        if ch == '\x1b':
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")

def draw_board(game, message=""):
    """Draw the game board in the terminal."""
    clear_screen()

    print("=" * 30)
    print("         2048 GAME")
    print("=" * 30)
    print(f"Score: {game.get_score()}")
    print()

    # Top border
    print("┌" + "─" * 25 + "┐")

    for row in game.grid:
        # Draw row
        line = "│"
        for power in row:
            if power == 0:
                line += "     "
            else:
                value = 2 ** power
                line += f"{value:>5}"
            line += " "
        line = line.rstrip() + " │"
        print(line)

    # Bottom border
    print("└" + "─" * 25 + "┘")
    print()

    if message:
        print(message)

    print("\nControls:")
    print("  ↑/W: Up    ↓/S: Down")
    print("  ←/A: Left  →/D: Right")
    print("  R: Restart  Q: Quit")

def main():
    """Main game loop."""
    game = Game2048()
    game.reset()
    game_over = False

    draw_board(game, "Welcome! Use arrow keys or WASD to play.")

    while True:
        key = get_key()

        # Handle quit
        if key.lower() == 'q':
            clear_screen()
            print("Thanks for playing!")
            break

        # Handle restart
        if key.lower() == 'r':
            game.reset()
            game_over = False
            draw_board(game, "Game restarted!")
            continue

        if game_over:
            draw_board(game, "GAME OVER! Press R to restart or Q to quit.")
            continue

        # Map keys to directions
        direction = None
        if key == '\x1b[A' or key.lower() == 'w':  # Up arrow or W
            direction = Direction.UP
        elif key == '\x1b[B' or key.lower() == 's':  # Down arrow or S
            direction = Direction.DOWN
        elif key == '\x1b[D' or key.lower() == 'a':  # Left arrow or A
            direction = Direction.LEFT
        elif key == '\x1b[C' or key.lower() == 'd':  # Right arrow or D
            direction = Direction.RIGHT

        if direction:
            _, reward, done, info = game.step(direction)

            if info.get("invalid_move"):
                draw_board(game, "Invalid move! Try another direction.")
            elif done:
                game_over = True
                draw_board(game, f"GAME OVER! Final Score: {game.get_score()}")
            else:
                message = f"Last move reward: {reward}" if reward > 0 else ""
                draw_board(game, message)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print("\nGame interrupted. Goodbye!")
    except Exception as e:
        clear_screen()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
