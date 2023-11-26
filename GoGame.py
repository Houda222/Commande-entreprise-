from processing import *
from mySgfCopy import GoSgf, GoVisual
class GoGame:

    def __init__(self, model):
        self.history = []
        self.moves = []
        self.current_state = []
        self.sgf = GoSgf("p1", "p2")
        self.model = model
        self.visual = GoVisual(self.sgf)


        