import numpy as np


class LaneLineHistory:
    def __init__(self, max_history=24):
        self.max_history = max_history
        self.left_lines = []
        self.right_lines = []

    def add_line(self, left_line, right_line):
        if left_line is not None:
            self.left_lines.append(left_line)
            if len(self.left_lines) > self.max_history:
                self.left_lines.pop(0)

        if right_line is not None:
            self.right_lines.append(right_line)
            if len(self.right_lines) > self.max_history:
                self.right_lines.pop(0)

    def get_average_line(self):
        left_line_avg = None
        right_line_avg = None

        if self.left_lines:
            left_line_avg = np.mean(self.left_lines, axis=0)
        if self.right_lines:
            right_line_avg = np.mean(self.right_lines, axis=0)

        return left_line_avg, right_line_avg

    def reset_history(self):
        self.left_lines = []
        self.right_lines = []
