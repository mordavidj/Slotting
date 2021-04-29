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
    min: int
    max: int

    def __init__(self, **kwargs):
        self.id = None
        self.desc = None
        self.case_qty = np.nan
        self.height = np.nan
        self.width = np.nan
        self.length = np.nan
        self.min = np.nan
        self.max = np.nan

        self.id = self.set_id(kwargs.get('id'))


    def display(self):
        '''Print the info in a pretty format

        '''
        print('')
        print(f'ID...........{self.id}')
        print(f'Description..{self.desc}')
        print(f'Case Qty.....{self.case_qty}')
        print('WxLxH........{0:.2} x {1:.2} x {2:.2}'.format(self.width, self.length, self.height))
        print(f'Min-Max......{self.min}-{self.max}')

    def get_info(self):
        '''Return the item's info in a dictionary

        '''
        return {'id'        : self.id, 
                'desc'      : self.desc, 
                'case_qty'  : self.case_qty, 
                'width'    : self.height, 
                'length'     : self.width, 
                'height'    : self.length,
                'min'       : self.min,
                'max'       : self.max}


    def get_dimensions(self):
        '''Return the item's box dimensions in a 3-part tuple

        '''
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
            #raise Exception('Case quantity must be an integer')
            return

        if case_qty > 0:
            self.case_qty = case_qty
        else:
            self.case_qty = np.nan
            #raise Exception('Case quantity cannot be less than 0')
            return


    def set_height(self, height):
        try:
            height = float(height)
        except:
            #raise Exception('Item height must be a number')
            return

        if height > 0:
            self.height = height
        else:
            self.height = np.nan
            #raise Exception('Item height cannot be less than 0') 
            return


    def set_width(self, width):
        try:
            width = float(width)
        except:
            #raise Exception('Item width must be a number')
            return

        if width > 0:
            self.width = width
        else:
            self.width = np.nan
            #raise Exception('Item width cannot be less than 0')
            return


    def set_length(self, length):
        try:
            length = float(length)
        except:
            #raise Exception('Item length must be a number')
            return

        if length > 0:
            self.length = length
        else:
            self.length = np.nan
            #raise Exception('Item length cannot be less than 0')
            return
    
    def get_minmax(self):
        return (self.min, self.max)


    def set_minmax(self, min, max):
        try:
            min = int(min)
            max = int(max)
        except:
            #raise Exception('Min and Max must be numbers')
            return
        
        if min > 0 and max >= min:
            self.min = min
            self.max = max
        else:
            self.min = np.nan
            self.max = np.nan
            #raise Exception('Min must be greater than 0 and Max greater than or equal to Min')
            return