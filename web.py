
from nicegui import ui 






# Store references to grid labels
grid_labels = []

with ui.grid(columns=4):
    for i in range(1, 18):
        label = ui.label(str(2**i))
        grid_labels.append(label)

# Track last action and debounce state
last_action = ui.label("Last action: None")
debounce_timer = None

def apply_grid_action(direction):
    """Apply increment/decrement action to grid numbers"""
    global debounce_timer
    
    # Cancel previous debounce timer if exists
    if debounce_timer is not None:
        debounce_timer.cancel()
    
    # Update last action display
    last_action.set_text(f"Last action: {direction}")
    
    # Debounced action (executes after 200ms)
    def execute_action():
        # Update all labels in the grid
        for label in grid_labels:
            try:
                current_value = int(label.text)
                if direction == "UP":
                    label.set_text(str(current_value + 1))
                elif direction == "DOWN":
                    label.set_text(str(current_value - 1))
            except (ValueError, AttributeError):
                pass
    
    # Set debounce timer
    from threading import Timer
    debounce_timer = Timer(0.2, execute_action)
    debounce_timer.start()

def handle_keyboard(e):
    """Handle keyboard arrow key events"""
    key = e.key
    if key == "ArrowUp":
        apply_grid_action("UP")
    elif key == "ArrowDown":
        apply_grid_action("DOWN")
    elif key == "ArrowLeft":
        last_action.set_text("Last action: LEFT (no effect)")
    elif key == "ArrowRight":
        last_action.set_text("Last action: RIGHT (no effect)")

# Register keyboard event handler
ui.keyboard(on_key=handle_keyboard)

ui.label("hello world")
ui.run()
