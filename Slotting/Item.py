class Item():
    def __init__(self):
        self.id = None
        self.name = None
        self.desc = None
        self.qty = None
        self.height = None
        self.width = None
        self.length = None

    def get_info(self):
        return {'id': self.id, 
                'name': self.name, 
                'desc': self.desc, 
                'qty': self.qty, 
                'height': self.height, 
                'width': self.width, 
                'length': self.length}

    def get_dimensions(self):
        return (self.width, self.length, self.height)