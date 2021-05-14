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


    def __init__(self, id: str = 'ID', description: str = '', case_qty: int = 0, 
                 height: float = 0.0, width: float = 0.0, length: float = 0.0, 
                 min: int = 0, max: int = 0, **kwargs):

        self.id = id
        self.desc = description
        self.case_qty = case_qty
        self.height = height
        self.width = width
        self.length = length
        self.min = min
        self.max = max



    def __str__(self):
        '''Return the item as string.

        '''
        return f'ID...........{self.id}' + \
               f'\nDescription..{self.desc}' + \
               f'\nCase Qty.....{self.case_qty}' + \
                '\nWxLxH........{0:.2f} x {1:.2f} x {2:.2f}'\
                .format(self.width, self.length, self.height) + \
               f'\nMin-Max......{self.min}-{self.max}'



    def display(self):
        '''Print the info in a pretty format.

        '''
        print('')
        print(f'ID...........{self.id}')
        print(f'Description..{self.desc}')
        print(f'Case Qty.....{self.case_qty}')
        print('WxLxH........{0:.2f} x {1:.2f} x {2:.2f}'\
            .format(self.width, self.length, self.height))
        print(f'Min-Max......{self.min}-{self.max}')



    def get_info(self):
        '''Return the item's info in a dictionary.

        '''
        return {'id'        : self.id, 
                'desc'      : self.desc, 
                'case_qty'  : self.case_qty, 
                'width'     : self.height, 
                'length'    : self.width, 
                'height'    : self.length,
                'min'       : self.min,
                'max'       : self.max}



    def get_dimensions(self):
        '''Return the item's box dimensions in a tuple:
            (width, length, height)

        '''
        return (self.width, self.length, self.height)



    def set_dimensions(self, width: float, length: float, height: float):
        '''Set the width, length, and height of the master carton box in inches for the item.

        '''
        self.set_width(width)
        self.set_length(length)
        self.set_height(height)



    def set_id(self, id: str):
        '''Set the item ID as a string.

        '''
        self.id = id



    def set_desc(self, desc: str):
        '''Set the item description as a string.

        '''
        self.desc = desc



    def set_case_qty(self, case_qty: int):
        '''Set the master carton case quantity of the item as an integer.

        '''
        if case_qty > 0:
            self.case_qty = case_qty
        else:
            print(f'item ({self.id}): Case quantity cannot be less than 0')



    def set_height(self, height: float):
        '''Set the height of the master carton case of the item in inches as an integer.

        '''
        if height > 0:
            self.height = height
        else:
            print(f'item ({self.id}): Height cannot be less than 0') 
            


    def set_width(self, width: float):
        '''Set the width of the master carton case of the item in inches as an integer.

        '''
        if width > 0:
            self.width = width
        else:
            print(f'item ({self.id}): Width cannot be less than 0')



    def set_length(self, length: float):
        '''Set the length of the master carton case of the item in inches as an integer.

        '''
        if length > 0:
            self.length = length
        else:
            print(f'item ({self.id}): Length cannot be less than 0')
    


    def get_minmax(self):
        '''Returns the min and max of the item in a tuple:
            (min, max)

        '''
        return (self.min, self.max)



    def set_minmax(self, min: int, max: int):
        '''Set the minimum and maximum of the item as a float.

        '''
        if min > 0 and max >= min:
            self.min = min
            self.max = max
        else:
            print(f'item ({self.id}): Min must be greater than 0 and Max greater than or equal to Min')