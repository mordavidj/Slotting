import numpy as np

class Item():
    '''Class to store item information for use within a pickface

    '''
    id: str
    desc: str
    case_qty: int
    height: float
    width: float
    length: float

    def __init__(self, **kwargs):
        self.id = None
        self.desc = None
        self.case_qty = np.nan
        self.height = np.nan
        self.width = np.nan
        self.length = np.nan

        self.id = self.set_id(kwargs.get('id'))


    def display(self):
        print('')
        print(f'ID...........{self.id}')
        print(f'Description..{self.desc}')
        print(f'Case Qty.....{self.case_qty}')
        print('WxLxH........{0:.2} x {1:.2} x {2:.2}'.format(self.width, self.length, self.height))

    def get_info(self):
        return {'id'        : self.id, 
                'desc'      : self.desc, 
                'case_qty'  : self.case_qty, 
                'height'    : self.height, 
                'width'     : self.width, 
                'length'    : self.length}


    def get_dimensions(self):
        return (self.width, self.length, self.height)


    def set_dimensions(self, width, length, height):
        self.set_width(width)
        self.set_length(length)
        self.set_height(height)


    def set_id(self, id):
        self.id = str(id)


    def set_desc(self, desc):
        self.desc = str(desc)


    def set_case_qty(self, case_qty):
        try:
            case_qty = int(case_qty)
        except:
            raise Exception('Case quantity must be an integer')

        if case_qty > 0:
            self.case_qty = case_qty
        else:
            self.case_qty = np.nan
            raise Exception('Case quantity cannot be less than 0')


    def set_height(self, height):
        try:
            height = float(height)
        except:
            raise Exception('Item height must be a number')

        if height > 0:
            self.height = height
        else:
            self.height = np.nan
            raise Exception('Item height cannot be less than 0') 


    def set_width(self, width):
        try:
            width = float(width)
        except:
            raise Exception('Item width must be a number')

        if width > 0:
            self.width = width
        else:
            self.width = np.nan
            raise Exception('Item width cannot be less than 0')


    def set_length(self, length):
        try:
            length = float(length)
        except:
            raise Exception('Item length must be a number')

        if length > 0:
            self.length = length
        else:
            self.length = np.nan
            raise Exception('Item length cannot be less than 0')