import pygame
import re
import itertools


class FourPlayerChessInterface:
    def __init__(self):
        pygame.init()

        self.board_width, self.board_height = 800, 800
        self.control_width = 200
        self.width = self.board_width + self.control_width
        self.height = self.board_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Four Player Chess Reviewer')

        self.square_size = self.board_width // 14

        self.current_game = 0
        self.current_index = 0
        self.square_size = self.height // 14
        self.light_square = (253, 246, 227)
        self.dark_square = (14, 110, 99)

        self.piece_images = {}
        self.load_piece_images()
        self.resize_piece_images()

        self.font = pygame.font.Font(None, 32)
        self.text_input = ""

        self.show_arrows = False
        self.min_confidence = 0.1

        self.dragging_slider = False
        self.text_input_focus = False

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

    def draw_slider(self):
        pygame.draw.rect(self.screen, (200, 200, 200), (self.board_width + 10, 50, 100, 10))
        pygame.draw.circle(self.screen, (0, 0, 0), (self.board_width + 10 + int(self.min_confidence * 100), 55), 5)
        slider_value_text = self.font.render(f"{self.min_confidence:.2f}", True, (0, 0, 0))
        self.screen.blit(slider_value_text, (self.board_width + 120, 45))

    def draw_arrows(self, move_probs, board):
        for i, prob in enumerate(move_probs.cpu().numpy()):
            if prob >= self.min_confidence:
                move = board.index_to_move(i)
                start_pos = (
                    move.From().GetCol() * self.square_size + self.square_size // 2,
                    move.From().GetRow() * self.square_size + self.square_size // 2
                )
                end_pos = (
                    move.To().GetCol() * self.square_size + self.square_size // 2,
                    move.To().GetRow() * self.square_size + self.square_size // 2
                )

                alpha = min(int(255 * prob * 1.25) + 20, 255)

                arrow_surface = pygame.Surface((self.board_width, self.board_height), pygame.SRCALPHA)

                # Draw the arrowhead
                direction = pygame.math.Vector2(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]).normalize()
                arrowhead_size = 21
                arrowhead_base_offset = arrowhead_size * 0.75  # Adjust this value to control where the arrow body ends
                adjusted_end_pos = (
                    end_pos[0] - direction.x * arrowhead_base_offset,
                    end_pos[1] - direction.y * arrowhead_base_offset
                )

                pygame.draw.line(arrow_surface, (255, 0, 0, alpha), start_pos, adjusted_end_pos, 5)

                left = direction.rotate(160) * arrowhead_size
                right = direction.rotate(-160) * arrowhead_size
                pygame.draw.polygon(arrow_surface, (255, 0, 0, alpha),
                                    [end_pos, (end_pos[0] + left.x, end_pos[1] + left.y),
                                     (end_pos[0] + right.x, end_pos[1] + right.y)])

                self.screen.blit(arrow_surface, (0, 0))

                # Display the probability at the base of the arrow
                base_offset = -10
                base_pos = (
                    start_pos[0] + direction.x * base_offset,
                    start_pos[1] + direction.y * base_offset
                )
                prob_text = f"{prob:.2f}"
                small_font = pygame.font.Font(None, 21)
                prob_surface = small_font.render(prob_text, True, (0, 200, 0))
                prob_rect = prob_surface.get_rect(center=base_pos)
                self.screen.blit(prob_surface, prob_rect.topleft)

    def handle_mouse_events(self, x, y, board):
        input_box_width = 100
        input_box_height = 40
        input_box_x = self.width - input_box_width - 10
        input_box_y = self.height - input_box_height - 50

        # Adjust the positions for the controls on the right side
        if self.board_width + 50 <= x <= self.board_width + 100 and 10 <= y <= 40:
            self.current_game = max(self.current_game - 1, 0)
            self.current_index = 0
        elif self.board_width + 100 <= x <= self.width - 50 and 10 <= y <= 40:
            self.current_game = min(self.current_game + 1, len(board.game_records) - 1)
            self.current_index = 0
        elif self.board_width + 10 <= x <= self.board_width + 110 and 50 <= y <= 60:
            self.min_confidence = (x - self.board_width - 10) / 100
        elif self.board_width + 10 <= x <= self.board_width + 40 and 10 <= y <= 40:  # Coordinates for the checkbox
            self.show_arrows = not self.show_arrows  # Toggle the checkbox state
        elif input_box_x <= x <= input_box_x + input_box_width and input_box_y <= y <= input_box_y + input_box_height:
            self.text_input_focus = True
        else:
            self.text_input_focus = False

    def handle_keyboard_events(self, event, board):
        if self.text_input_focus:
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
                # Capture alphanumeric keys and some special characters
                valid_characters = "0123456789"
                if event.unicode in valid_characters:
                    self.text_input += event.unicode
        else:
            if event.key == pygame.K_LEFT:
                self.current_index = max(self.current_index - 1, 0)
                if self.current_index == 0:
                    self.current_game = max(self.current_game - 1, 0)
            elif event.key == pygame.K_RIGHT:
                self.current_index += 1
                if self.current_index >= len(board.game_records[self.current_game]):
                    self.current_index = 0
                    self.current_game = min(self.current_game + 1, len(board.game_records) - 1)

    def draw_board_state(self, board_state):
        """Draws the board state based on the current game state information."""
        self.screen.fill((255, 255, 255))

        for x in range(14):
            for y in range(14):
                if (x < 3 and y < 3) or (x >= 11 and y < 3) or (x < 3 and y >= 11) or (x >= 11 and y >= 11):
                    continue
                square_color = self.light_square if (x + y) % 2 == 0 else self.dark_square
                pygame.draw.rect(self.screen, square_color,
                                 (x * self.square_size, y * self.square_size, self.square_size, self.square_size))

        # Assuming game_state provides access to piece information similar to the original method
        for piece_info in list(itertools.chain(*board_state.GetPieces())):
            color, piece_type, position = self.parse_piece_info(str(piece_info))
            self.draw_piece(color, piece_type, position)

        pygame.display.flip()

    def parse_piece_info(self, piece_str):
        """Parse piece information to extract color, type, and board position."""
        piece_pattern = re.compile(r"(\w+) (\w+) at (\w\d+)")
        match = piece_pattern.match(piece_str)
        if match:
            color, piece_type, position = match.groups()
            return color, piece_type, position
        return None, None, None

    def draw_piece(self, color, piece_type, uci_position):
        """Draws a piece on the board based on its type and position."""
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
        """Convert UCI position string to board coordinates (row, col)."""
        col = ord(uci_position[0]) - ord('a')
        row = 14 - int(uci_position[1:])
        return row, col

    def run(self, board):
        pygame.init()

        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    self.handle_mouse_events(x, y, board)

                if event.type == pygame.KEYDOWN:
                    self.handle_keyboard_events(event, board)

            self.screen.fill((255, 255, 255))
            self.draw_toggle_button()
            self.draw_slider()

            # Draw the current game index
            game_index_text = self.font.render(f"Game: {self.current_game + 1}/{len(board.game_records)}", True,
                                               (0, 0, 0))
            game_index_position = (
            self.width - game_index_text.get_width() - 10, self.height - game_index_text.get_height() - 10)
            self.screen.blit(game_index_text, game_index_position)

            input_box_width = 100
            input_box_height = 40
            input_box_x = self.width - input_box_width - 10
            input_box_y = game_index_position[1] - input_box_height - 5
            pygame.draw.rect(self.screen, (0, 0, 0), (input_box_x, input_box_y, input_box_width, input_box_height), 2)
            text_surface = self.font.render(self.text_input, True, (0, 0, 0))
            self.screen.blit(text_surface, (input_box_x + 10, input_box_y + 5))

            if self.current_game >= len(board.game_records):
                raise IndexError(f"Game index {self.current_game} is out of bounds")

            board_state = board.game_records[self.current_game][self.current_index]
            self.draw_board_state(board_state)

            if self.show_arrows:
                self.draw_arrows(board_state['move_probs'], board)

            pygame.display.flip()

        pygame.quit()
