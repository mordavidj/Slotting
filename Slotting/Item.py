class Item():
    def __init__(self):
        self.id = None
        self.name = None
        self.desc = None
        self.case_qty = None
        self.height = None
        self.width = None
        self.length = None

    def display(self):
        print(f'ID...........{self.id}')
        print(f'Name.........{self.name}')
        print(f'Description..{self.desc}')
        print(f'Case Qty.....{self.case_qty}')
        print(f'WxLxH........{self.width}x{self.length}x{self.height}')

    def get_info(self):
        return {'id'        : self.id, 
                'name'      : self.name, 
                'desc'      : self.desc, 
                'case_qty'  : self.case_qty, 
                'height'    : self.height, 
                'width'     : self.width, 
                'length'    : self.length}


    def get_dimensions(self):
        return (self.width, self.length, self.height)


    def set_id(self, id):
        self.id = str(id)


    def set_name(self, name):
        self.name = str(name)


    def set_desc(self, desc):
        self.desc = str(desc)


    def set_case_qty(self, case_qty):
        try:
            case_qty = int(case_qty)
        except:
            raise Exception('Item quantity must be an integer')

        if case_qty < 0:
            self.case_qty = case_qty
        else:
            raise Exception('Item quantity cannot be less than 0')


    def set_height(self, height):
        try:
            height = float(height)
        except:
            raise Exception('Item height must be a number')

        if height < 0:
            self.height = height
        else:
            raise Exception('Item height cannot be less than 0') 


    def set_width(self, width):
        try:
            width = float(width)
        except:
            raise Exception('Item width must be a number')

        if width < 0:
            self.width = width
        else:
            raise Exception('Item width cannot be less than 0')


    def set_length(self, length):
        try:
            length = float(length)
        except:
            raise Exception('Item length must be a number')

        if length < 0:
            self.length = length
        else:
            raise Exception('Item length cannot be less than 0')