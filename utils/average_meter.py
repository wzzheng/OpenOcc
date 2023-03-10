
class AverageMeter:

    def __init__(self) -> None:
        
        self.val = 0
        self.cnt = 0
        self.sum = 0

    @property
    def avg(self):
        return self.sum / self.cnt
    
    def update(self, val):
        self.val = val
        self.sum += val
        self.cnt += 1

    def reset(self):
        self.val = 0
        self.cnt = 0
        self.sum = 0