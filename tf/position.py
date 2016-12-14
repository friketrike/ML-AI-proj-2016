import board as b


class Pos:
    def __init__(self, horizontal, vertical):
            self.h = horizontal
            self.v = vertical

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        hash((tuple(self.h), tuple(self.v)))

    @classmethod
    def from_string(cls, string):
        h = ord(string[0]) - ord('a')
        v = ord(string[2]) - ord('1')
        pos = cls(h, v)
        if not pos.is_valid():
            raise ValueError
        return pos

    def is_valid(self):
        return (0 <= self.h < b.BOARD_SIZE and 0 <= self.v < b.BOARD_SIZE )

    def to_string(self):
        horizontal = chr(ord('a') + self.h)
        vertical = chr(ord('1') + self.v)
        return horizontal + ', ' + vertical

    @staticmethod
    def down(pos):
        if pos.v < b.BOARD_SIZE-1:
            pos.v += 1
            return True
        else:
            return False

    @staticmethod
    def right(pos):
        if pos.h < b.BOARD_SIZE-1:
            pos.h += 1
            return True
        else:
            return False

    @staticmethod
    def up(pos):
        if pos.v > 0:
            pos.v -= 1
            return True
        else:
            return False

    @staticmethod
    def left(pos):
        if pos.h > 0:
            pos.h -= 1
            return True
        else:
            return False

    @staticmethod
    def down_right(pos):
        if (pos.h < b.BOARD_SIZE-1) and (pos.v < b.BOARD_SIZE-1):
            pos.h += 1
            pos.v += 1
            return True
        else:
            return False

    @staticmethod
    def down_left(pos):
        if (pos.h > 0) and (pos.v < b.BOARD_SIZE-1):
            pos.h -= 1
            pos.v += 1
            return True
        else:
            return False

    @staticmethod
    def up_right(pos):
        if (pos.h < b.BOARD_SIZE-1) and (pos.v > 0):
            pos.h += 1
            pos.v -= 1
            return True
        else:
            return False

    @staticmethod
    def up_left(pos):
        if (pos.h > 0) and (pos.v > 0):
            pos.h -= 1
            pos.v -= 1
            return True
        else:
            return False


