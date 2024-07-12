import pygame
import re
import itertools
from four_player_chess_board import FourPlayerChess

class FourPlayerChessInterface:
    def __init__(self, queue):
        pygame.init()

        self.board_width, self.board_height = 600, 600
        self.control_width = 200
        self.width = self.board_width + self.control_width
        self.height = self.board_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Four Player Chess Reviewer')

        self.square_size = self.board_width // FourPlayerChess.nCols()
        self.light_square = (253, 246, 227)
        self.dark_square = (15, 110, 99)

        self.piece_images = {}
        self.load_piece_images()
        self.resize_piece_images()

        self.font = pygame.font.Font(None, 32)
        self.text_input = ""

        self.show_arrows = False
        self.min_confidence = 0.1

        self.dragging_slider = False
        self.text_input_focus = False

        self.current_batch = 0
        self.current_game = 0
        self.current_index = 0
        self.queue = queue

        self.clock = pygame.time.Clock()
        self.frame_rate = 30  # Limit the frame rate to 30 FPS

        self.batches = []

    def load_piece_images(self):
        colors = ["blue", "green", "red", "yellow"]
        pieces = ["king", "queen", "rook", "bishop", "knight", "pawn"]

        for color in colors:
            for piece in pieces:
                try:
                    self.piece_images[f"{color}_{piece}"] = pygame.image.load(f'./assets/{color}_{piece}.png')
                except pygame.error:
                    print(f"Image not found for: {color}_{piece}.png")

    def resize_piece_images(self):
        self.pygame_piece_images = {}
        for key, img in self.piece_images.items():
            if "pawn" in key:
                resize_scale = 0.7
            else:
                resize_scale = 0.8

            new_size = int(self.square_size * resize_scale)
            resized_img = pygame.transform.scale(img, (new_size, new_size))
            self.pygame_piece_images[key] = resized_img

    def draw_board_state(self, board_state):
        for x in range(FourPlayerChess.nCols()):
            for y in range(FourPlayerChess.nRows()):
                if not FourPlayerChess.IsLegalLocation(x, y):
                    continue
                square_color = self.light_square if (x + y) % 2 == 0 else self.dark_square
                pygame.draw.rect(self.screen, square_color,
                                 (x * self.square_size, y * self.square_size, self.square_size, self.square_size))

        for piece_info in board_state:
            color, piece_type, position = self.parse_piece_info(piece_info)
            self.draw_piece(color, piece_type, position)

    @staticmethod
    def parse_piece_info(piece_str):
        piece_pattern = re.compile(r"(\w+) (\w+) at (\w\d+)")
        match = piece_pattern.match(piece_str)
        if match:
            color, piece_type, position = match.groups()
            return color, piece_type, position
        return None, None, None

    def draw_piece(self, color, piece_type, uci_position):
        if color and piece_type and uci_position:
            row, col = self._uci_to_location(uci_position)
            piece_key = f"{color.lower()}_{piece_type.lower()}"
            img = self.pygame_piece_images.get(piece_key)
            if img:
                offset_x = (self.square_size - img.get_width()) // 2
                offset_y = (self.square_size - img.get_height()) // 2
                screen_x = col * self.square_size + offset_x
                screen_y = row * self.square_size + offset_y
                self.screen.blit(img, (screen_x, screen_y))

    def _uci_to_location(self, uci_position):
        col = ord(uci_position[0]) - ord('a')
        row = FourPlayerChess.nRows() - int(uci_position[1:])
        return row, col

    def draw_controls(self):
        self.draw_toggle_button()

        total_batches = len(self.batches)
        total_games = len(self.batches[self.current_batch]) if total_batches > 0 else 0
        total_states = len(self.batches[self.current_batch][self.current_game]) if total_games > 0 else 0

        batch_text = self.font.render(f"Batch: {self.current_batch + 1}/{total_batches}", True, (0, 0, 0))
        game_text = self.font.render(f"Game: {self.current_game + 1}/{total_games}", True, (0, 0, 0))
        state_text = self.font.render(f"State: {self.current_index + 1}/{total_states}", True, (0, 0, 0))

        self.screen.blit(batch_text, (self.board_width + 10, 70))
        self.screen.blit(game_text, (self.board_width + 10, 100))
        self.screen.blit(state_text, (self.board_width + 10, 130))

    def draw_toggle_button(self):
        checkbox_x = self.board_width + 10
        checkbox_y = 10
        checkbox_size = 30

        pygame.draw.rect(self.screen, (200, 200, 200), (checkbox_x, checkbox_y, checkbox_size, checkbox_size))

        if self.show_arrows:
            inner_margin = 5
            pygame.draw.rect(self.screen, (0, 0, 0), (
                checkbox_x + inner_margin, checkbox_y + inner_margin, checkbox_size - 2 * inner_margin,
                checkbox_size - 2 * inner_margin))

        button_text = self.font.render("Show Arrows", True, (0, 0, 0))
        self.screen.blit(button_text, (self.board_width + 50, 10))

    def handle_mouse_events(self, x, y):
        if self.board_width + 10 <= x <= self.board_width + 40 and 10 <= y <= 40:
            self.show_arrows = not self.show_arrows
        else:
            self.text_input_focus = False

    def handle_keyboard_events(self, event):
        if event.key == pygame.K_RETURN:
            try:
                self.current_game = int(self.text_input) - 1
                self.current_index = 0
                self.text_input = ""
            except ValueError:
                print("Invalid game index")
        elif event.key == pygame.K_BACKSPACE:
            self.text_input = self.text_input[:-1]
        else:
            valid_characters = "0123456789"
            if event.unicode in valid_characters:
                self.text_input += event.unicode
        if event.key == pygame.K_LEFT:
            self.current_index = max(self.current_index - 1, 0)
            if self.current_index == 0:
                self.current_game = max(self.current_game - 1, 0)
        elif event.key == pygame.K_RIGHT:
            self.current_index += 1
            if self.batches:
                total_games = len(self.batches[self.current_batch])
                if total_games:
                    total_states = len(self.batches[self.current_batch][self.current_game])
                    if self.current_index >= total_states:
                        self.current_index = 0
                        self.current_game = min(self.current_game + 1, total_games - 1)
        elif event.key == pygame.K_UP:
            self.current_batch = max(self.current_batch - 1, 0)
            self.current_game = 0
            self.current_index = 0
        elif event.key == pygame.K_DOWN:
            self.current_batch = min(self.current_batch + 1, len(self.batches) - 1)
            self.current_game = 0
            self.current_index = 0

    def run(self):
        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    self.handle_mouse_events(x, y)

                if event.type == pygame.KEYDOWN:
                    self.handle_keyboard_events(event)

            while not self.queue.empty():
                message = self.queue.get()
                if message == "new batch":
                    self.batches.append([])
                else:
                    self.batches[-1].append(message)

            self.screen.fill((255, 255, 255))
            self.draw_controls()

            if self.batches and len(self.batches[self.current_batch]) > 0:
                batch = self.batches[self.current_batch]
                game = batch[self.current_game]
                state = game[self.current_index]

                self.draw_board_state(state)  # state is the serialized representation of the board state

            pygame.display.flip()
            self.clock.tick(self.frame_rate)  # Limit the frame rate

        pygame.quit()

def run_ui(queue):
    ui = FourPlayerChessInterface(queue)
    ui.run()
