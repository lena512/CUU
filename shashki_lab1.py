import tkinter as tk
from tkinter import messagebox
from enum import Enum
from typing import Optional, List, Tuple, Set


class Player(Enum):
    WHITE = "white"      # первый игрок (ходит снизу вверх)
    BLACK = "black"      # второй игрок (ходит сверху вниз)

    def opposite(self) -> "Player":
        return Player.BLACK if self == Player.WHITE else Player.WHITE


class Piece:
    """Игровая фигура (шашка/дамка)"""
    def __init__(self, player: Player, row: int, col: int):
        self.player = player
        self.row = row
        self.col = col
        self.is_king = False

    def make_king(self):
        self.is_king = True


class Board:
    """Логическое представление доски 8x8"""
    SIZE = 8

    def __init__(self):
        self.grid: List[List[Optional[Piece]]] = [[None for _ in range(self.SIZE)] for _ in range(self.SIZE)]

    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        if 0 <= row < self.SIZE and 0 <= col < self.SIZE:
            return self.grid[row][col]
        return None

    def set_piece(self, row: int, col: int, piece: Optional[Piece]):
        if 0 <= row < self.SIZE and 0 <= col < self.SIZE:
            self.grid[row][col] = piece
            if piece:
                piece.row, piece.col = row, col

    def remove_piece(self, row: int, col: int) -> Optional[Piece]:
        piece = self.get_piece(row, col)
        self.set_piece(row, col, None)
        return piece

    def is_empty(self, row: int, col: int) -> bool:
        return self.get_piece(row, col) is None

    def is_valid_cell(self, row: int, col: int) -> bool:
        return 0 <= row < self.SIZE and 0 <= col < self.SIZE

    @staticmethod
    def is_dark_cell(row: int, col: int) -> bool:
        """На шахматной доске фигуры стоят только на тёмных клетках"""
        return (row + col) % 2 == 1


class GameRules:
    """Правила шашек (русские шашки)"""
    @staticmethod
    def is_opponent(piece1: Optional[Piece], piece2: Optional[Piece]) -> bool:
        return piece1 is not None and piece2 is not None and piece1.player != piece2.player

    @staticmethod
    def get_normal_moves(board: Board, piece: Piece) -> List[Tuple[int, int]]:
        """Возвращает все возможные обычные ходы (без взятия) для простой шашки"""
        moves = []
        row, col = piece.row, piece.col
        direction = -1 if piece.player == Player.WHITE else 1  # белые ходят вверх, чёрные вниз

        for dc in (-1, 1):
            new_row, new_col = row + direction, col + dc
            if board.is_valid_cell(new_row, new_col) and board.is_empty(new_row, new_col):
                moves.append((new_row, new_col))
        return moves

    @staticmethod
    def get_king_moves(board: Board, piece: Piece) -> List[Tuple[int, int]]:
        """Все возможные диагональные ходы дамки на пустые клетки"""
        moves = []
        row, col = piece.row, piece.col
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            r, c = row + dr, col + dc
            while board.is_valid_cell(r, c) and board.is_empty(r, c):
                moves.append((r, c))
                r += dr
                c += dc
        return moves

    @staticmethod
    def get_normal_captures(board: Board, piece: Piece) -> List[Tuple[int, int, int, int]]:
        """
        Возвращает список возможных взятий для простой шашки.
        Каждое взятие описывается кортежем (target_row, target_col, capture_row, capture_col)
        """
        captures = []
        row, col = piece.row, piece.col
        direction = -1 if piece.player == Player.WHITE else 1

        for dc in (-1, 1):
            mid_r, mid_c = row + direction, col + dc
            land_r, land_c = row + 2 * direction, col + 2 * dc
            if board.is_valid_cell(land_r, land_c) and board.is_empty(land_r, land_c):
                mid_piece = board.get_piece(mid_r, mid_c)
                if GameRules.is_opponent(piece, mid_piece):
                    captures.append((land_r, land_c, mid_r, mid_c))
        return captures

    @staticmethod
    def get_king_captures(board: Board, piece: Piece) -> List[Tuple[int, int, int, int]]:
        """
        Возвращает возможные взятия для дамки.
        Каждое взятие: (land_row, land_col, captured_row, captured_col)
        """
        captures = []
        row, col = piece.row, piece.col
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            r, c = row + dr, col + dc
            captured = None
            while board.is_valid_cell(r, c):
                current = board.get_piece(r, c)
                if current is not None:
                    if captured is None and GameRules.is_opponent(piece, current):
                        captured = (r, c, current)
                    else:
                        break
                elif captured is not None:
                    # За шашкой есть пустая клетка – можем бить
                    captures.append((r, c, captured[0], captured[1]))
                r += dr
                c += dc
        return captures

    @staticmethod
    def get_all_moves(board: Board, piece: Piece, mandatory_capture: bool = True) -> List[Tuple[int, int, Optional[Tuple[int, int]]]]:
        """
        Возвращает список всех допустимых ходов для фигуры.
        Каждый ход: (to_row, to_col, captured_pos) где captured_pos = (row, col) или None.
        """
        captures = GameRules.get_normal_captures(board, piece) if not piece.is_king else GameRules.get_king_captures(board, piece)
        if captures:
            return [(land_r, land_c, (cap_r, cap_c)) for land_r, land_c, cap_r, cap_c in captures]
        if mandatory_capture:
            return []
        moves = GameRules.get_normal_moves(board, piece) if not piece.is_king else GameRules.get_king_moves(board, piece)
        return [(r, c, None) for r, c in moves]

    @staticmethod
    def has_any_capture(board: Board, player: Player) -> bool:
        """Проверяет, есть ли у игрока хоть одно взятие"""
        for row in range(Board.SIZE):
            for col in range(Board.SIZE):
                piece = board.get_piece(row, col)
                if piece and piece.player == player:
                    captures = GameRules.get_normal_captures(board, piece) if not piece.is_king else GameRules.get_king_captures(board, piece)
                    if captures:
                        return True
        return False

    @staticmethod
    def get_all_player_moves(board: Board, player: Player) -> List[Tuple[Piece, int, int, Optional[Tuple[int, int]]]]:
        """Все ходы текущего игрока (фигура + целевая клетка + сбитая позиция)"""
        moves = []
        mandatory = GameRules.has_any_capture(board, player)
        for row in range(Board.SIZE):
            for col in range(Board.SIZE):
                piece = board.get_piece(row, col)
                if piece and piece.player == player:
                    for to_row, to_col, captured in GameRules.get_all_moves(board, piece, mandatory):
                        moves.append((piece, to_row, to_col, captured))
        return moves


class CheckersGameLogic:
    """Управление игрой, хранение состояния"""
    def __init__(self):
        self.board = Board()
        self.current_player = Player.WHITE
        self.pieces_count = {Player.WHITE: 0, Player.BLACK: 0}
        self.move_without_capture = 0   # для правила 15 ходов
        self.winner: Optional[Player] = None

        self._init_pieces()

    def _init_pieces(self):
        """Расстановка шашек в начальной позиции"""
        # Белые (первый игрок) снизу – строки 6,7? По правилам строки 5,6,7 (с 0)
        # Чёрные сверху – строки 0,1,2
        for row in range(Board.SIZE):
            for col in range(Board.SIZE):
                if Board.is_dark_cell(row, col):
                    if row < 3:
                        piece = Piece(Player.BLACK, row, col)
                        self.board.set_piece(row, col, piece)
                        self.pieces_count[Player.BLACK] += 1
                    elif row > 4:
                        piece = Piece(Player.WHITE, row, col)
                        self.board.set_piece(row, col, piece)
                        self.pieces_count[Player.WHITE] += 1

    def apply_move(self, piece: Piece, to_row: int, to_col: int, captured_pos: Optional[Tuple[int, int]]) -> bool:
        """Применяет ход, изменяет доску и состояние игры. Возвращает True, если ход успешен"""
        start_row, start_col = piece.row, piece.col

        # Проверка на превращение в дамку
        will_be_king = False
        if not piece.is_king:
            if piece.player == Player.WHITE and to_row == 0:
                will_be_king = True
            elif piece.player == Player.BLACK and to_row == Board.SIZE - 1:
                will_be_king = True

        # Перемещаем фигуру
        self.board.set_piece(start_row, start_col, None)
        self.board.set_piece(to_row, to_col, piece)
        if will_be_king:
            piece.make_king()

        # Обрабатываем взятие
        if captured_pos:
            cap_row, cap_col = captured_pos
            captured_piece = self.board.remove_piece(cap_row, cap_col)
            if captured_piece:
                self.pieces_count[captured_piece.player] -= 1
            self.move_without_capture = 0
        else:
            self.move_without_capture += 1

        # Проверка на продолжение боя (обязательное)
        # Если после хода фигура может бить дальше – не переключаем игрока
        if captured_pos:
            # Проверим, может ли эта же фигура продолжить взятие
            next_captures = GameRules.get_normal_captures(self.board, piece) if not piece.is_king else GameRules.get_king_captures(self.board, piece)
            if next_captures:
                return True  # не переключаем игрока, даём ход повторно

        # Переключение игрока
        self.current_player = self.current_player.opposite()

        # Проверка окончания игры
        self._check_game_over()
        return True

    def _check_game_over(self):
        """Проверяет, закончена ли игра (нет ходов у текущего игрока)"""
        moves = GameRules.get_all_player_moves(self.board, self.current_player)
        if not moves:
            self.winner = self.current_player.opposite()
        elif self.move_without_capture >= 15:
            # Ничья по правилу 15 ходов без взятия – победителя нет, но оба проиграли?
            # В классике ничья, упростим – победитель по оставшимся шашкам
            if self.pieces_count[Player.WHITE] > self.pieces_count[Player.BLACK]:
                self.winner = Player.WHITE
            elif self.pieces_count[Player.BLACK] > self.pieces_count[Player.WHITE]:
                self.winner = Player.BLACK
            else:
                self.winner = None  # ничья
        elif self.pieces_count[self.current_player] == 0:
            self.winner = self.current_player.opposite()

    def is_game_over(self) -> bool:
        return self.winner is not None

    def get_winner_name(self) -> str:
        if self.winner == Player.WHITE:
            return "Игрок 1"
        if self.winner == Player.BLACK:
            return "Игрок 2"
        return "Ничья"



class GameGUI:
    CELL_SIZE = 50

    def __init__(self, parent, logic: CheckersGameLogic):
        self.parent = parent
        self.logic = logic
        self.canvas = None
        self.info_label = None
        self.points_label = None
        self._selected_piece = None   # хранит выбранную фигуру

        self._build_ui()
        self._redraw_all()            # отрисовываем всё с нуля
        self._update_info()

    def _build_ui(self):
        # Очищаем родительский контейнер
        for widget in self.parent.winfo_children():
            widget.destroy()

        self.canvas = tk.Canvas(self.parent, width=Board.SIZE * self.CELL_SIZE,
                                height=Board.SIZE * self.CELL_SIZE, bg="white")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self._on_click)

        self.info_label = tk.Label(self.parent, text="", font=("Comic Sans MS", 15))
        self.info_label.pack()
        self.points_label = tk.Label(self.parent, text="", font=("Comic Sans MS", 35))
        self.points_label.pack()

        tk.Button(self.parent, text="Выйти в меню", font=("Comic Sans MS", 12),
                  command=self._quit_to_menu).pack(pady=5)

    def _redraw_all(self):
        """Полностью перерисовывает доску и все фигуры."""
        self.canvas.delete("all")          # удаляем всё с холста
        self._draw_board()
        self._draw_pieces()
        if hasattr(self, '_selected_piece') and self._selected_piece is not None:
            self._highlight_selected(self._selected_piece.row, self._selected_piece.col)

    def _draw_board(self):
        for row in range(Board.SIZE):
            for col in range(Board.SIZE):
                x1 = col * self.CELL_SIZE
                y1 = row * self.CELL_SIZE
                x2 = x1 + self.CELL_SIZE
                y2 = y1 + self.CELL_SIZE
                color = "#E0E0E0" if (row + col) % 2 == 0 else "#A0A0A0"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def _draw_pieces(self):
        """Рисует все фигуры на основе состояния логики."""
        for row in range(Board.SIZE):
            for col in range(Board.SIZE):
                piece = self.logic.board.get_piece(row, col)
                if piece:
                    x_center = col * self.CELL_SIZE + self.CELL_SIZE // 2
                    y_center = row * self.CELL_SIZE + self.CELL_SIZE // 2
                    radius = self.CELL_SIZE // 2 - 5
                    color = "white" if piece.player == Player.WHITE else "black"
                    # Рисуем круг
                    self.canvas.create_oval(x_center - radius, y_center - radius,
                                            x_center + radius, y_center + radius,
                                            fill=color, outline="gray", width=2, tags="piece")
                    # Если дамка – рисуем корону
                    if piece.is_king:
                        self.canvas.create_text(x_center, y_center, text="★",
                                                font=("Arial", 20, "bold"), fill="gold",
                                                tags="king")
        # Поднимаем выделение поверх фигур (если есть)
        if hasattr(self, '_selected_piece') and self._selected_piece is not None:
            self._highlight_selected(self._selected_piece.row, self._selected_piece.col)

    def _highlight_selected(self, row, col):
        x1 = col * self.CELL_SIZE
        y1 = row * self.CELL_SIZE
        x2 = x1 + self.CELL_SIZE
        y2 = y1 + self.CELL_SIZE
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=3, tags="highlight")

    def _clear_highlight(self):
        self.canvas.delete("highlight")

    def _on_click(self, event):
        if self.logic.is_game_over():
            return

        col = event.x // self.CELL_SIZE
        row = event.y // self.CELL_SIZE
        if not (0 <= row < Board.SIZE and 0 <= col < Board.SIZE):
            return

        clicked_piece = self.logic.board.get_piece(row, col)

        # Если ещё не выбрана фигура – пытаемся выбрать свою
        if self._selected_piece is None:
            if clicked_piece and clicked_piece.player == self.logic.current_player:
                self._selected_piece = clicked_piece
                self._redraw_all()    # перерисовываем, чтобы подсветить выбранную
            return

        # Уже выбрана фигура – пробуем сделать ход
        piece = self._selected_piece
        mandatory = GameRules.has_any_capture(self.logic.board, self.logic.current_player)
        moves = GameRules.get_all_moves(self.logic.board, piece, mandatory)

        for to_row, to_col, captured in moves:
            if to_row == row and to_col == col:
                success = self.logic.apply_move(piece, to_row, to_col, captured)
                if success:
                    self._selected_piece = None
                    self._redraw_all()
                    self._update_info()
                    if self.logic.is_game_over():
                        self._show_game_over()
                else:
                    self._selected_piece = None
                    self._redraw_all()
                return

        # Неверный ход – сбрасываем выделение
        self._selected_piece = None
        self._redraw_all()

    def _update_info(self):
        white_left = self.logic.pieces_count[Player.WHITE]
        black_left = self.logic.pieces_count[Player.BLACK]
        self.points_label.config(text=f"{white_left} : {black_left}")

        player_name = "Первый игрок (белые)" if self.logic.current_player == Player.WHITE else "Второй игрок (чёрные)"
        self.info_label.config(text=f"Ход: {player_name}")

    def _show_game_over(self):
        winner = self.logic.get_winner_name()
        messagebox.showinfo("Конец игры", f"Победил {winner}!")
        self._quit_to_menu()

    def _quit_to_menu(self):
        self.parent.destroy()
        root = tk.Tk()
        app = CheckersGame(root)
        root.mainloop()


class CheckersGame:
    """Главное приложение: логин/меню и запуск игры"""
    def __init__(self, master):
        self.master = master
        self.master.title("Шашки")
        self.master.geometry("500x300")
        self.master.resizable(False, False)
        self.login_frame = tk.Frame(master)
        self.menu_frame = tk.Frame(master)
        self.show_login()

    def show_login(self):
        self.menu_frame.pack_forget()
        self.login_frame.pack()

        tk.Label(self.login_frame, text="Шашки Артамонова", font=("Comic Sans MS", 22)).grid(row=0, column=0, columnspan=2, pady=10)
        tk.Label(self.login_frame, text="Логин:", font=("Comic Sans MS", 16)).grid(row=1, column=0, sticky="e")
        tk.Label(self.login_frame, text="Пароль:", font=("Comic Sans MS", 16)).grid(row=2, column=0, sticky="e")
        self.username_entry = tk.Entry(self.login_frame, width=25)
        self.password_entry = tk.Entry(self.login_frame, width=25, show="*")
        self.username_entry.grid(row=1, column=1, padx=10, pady=5)
        self.password_entry.grid(row=2, column=1, padx=10, pady=5)

        tk.Button(self.login_frame, text="Регистрация", font=("Comic Sans MS", 12),
                  command=self.register).grid(row=3, column=0, columnspan=2, pady=5)
        tk.Button(self.login_frame, text="Войти", font=("Comic Sans MS", 12),
                  command=self.login).grid(row=4, column=0, columnspan=2, pady=5)

    def login(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()
        if username and password:
            messagebox.showinfo("Вход", "Вы успешно вошли в систему!")
            self.show_start_menu()
        else:
            messagebox.showwarning("Ошибка", "Введите логин и пароль.")

    def register(self):
        messagebox.showinfo("Регистрация", "Регистрация выполнена (учебный пример).")

    def show_start_menu(self):
        self.login_frame.pack_forget()
        self.menu_frame.pack()

        tk.Label(self.menu_frame, text="Шашки Артамонова", font=("Comic Sans MS", 22)).pack(pady=20)
        tk.Button(self.menu_frame, text="Друг против друга", font=("Comic Sans MS", 14),
                  command=self.start_game).pack(fill=tk.X, pady=10, padx=40)
        tk.Button(self.menu_frame, text="Игра с ботом", font=("Comic Sans MS", 14),
                  command=self.start_game).pack(fill=tk.X, pady=10, padx=40)
        tk.Button(self.menu_frame, text="Выйти", font=("Comic Sans MS", 14),
                  command=self.master.quit).pack(fill=tk.X, pady=10, padx=40)

    def start_game(self):
        self.master.destroy()
        root = tk.Tk()
        logic = CheckersGameLogic()
        GameGUI(root, logic)
        root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = CheckersGame(root)
    root.mainloop()

