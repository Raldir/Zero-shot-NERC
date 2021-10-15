
class Token(object):
    """A single token object."""

    def __init__(self, word, type, type_id = None, pos = None, const = None):
        self.word = word
        self.pos = pos
        self.const = const
        self.type = type
        self.type_id = type_id

    def __str__(self):
        return '\n'.join(["Word: " + str(self.word), "POS:" + str(self.pos), "Constituent:" + str(self.const), "Entity type:" + str(self.type), "Entity type id:" + str(self.type_id)])

    def __repr__(self):
        return self.word
