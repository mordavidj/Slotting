import pandas as pd
pd.set_option('max_columns', None)
import math
from Item import *
import csv
import datetime

ROWS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']


class Pickface():
    '''Pickface class that stores info about the pickface and stores an array of items.

    '''
    bays: int 
    bay_cols: int
    bay_rows: int
    row_height: int
    depth: int
    slots: list 
    cust: str 
    row_priority: list
    col_priority: list

    def __init__(self):
        self.bays = 0
        self.bay_cols = 0
        self.bay_rows = 0
        self.row_height = 0
        self.depth = 0
        self.slots = []
        self.cust = ''
        self.row_priority = []
        self.col_priority = []


    def display(self):
        '''Print the PF to console

        '''
        print('')
        print(f'Customer: {self.cust}')
        print('|||||||||||||||||||||||||||||||||||||||||||||||')
        
        for b in range(self.bays):
            row_del = ''
            for r in range(self.bay_rows - 1, -1, -1):
                print(row_del) 
                for c in range(self.bay_cols):
                    print(f'## {str(b+1).zfill(2)}.{ROWS[r]}{str(c+1).zfill(2)}: {self.slots[b][r][c].id} ##', end='')

                row_del = '\n-----------------------------------------------'

            print('\n|||||||||||||||||||||||||||||||||||||||||||||||')


    def list_items(self):
        '''Return all the items from the pickface

        '''
        items = []

        for b in range(self.bays):
            for r in range(self.bay_rows):
                for c in range(self.bay_cols):
                    items.append(self.slots[b][r][c].id)

        return items


    def to_csv(self):
        '''write the pickface to an excel file

        '''
        with open(f'data/{self.cust}-{self.bays * self.bay_cols * self.bay_rows}.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|')
            writer.writerow(['Customer:', self.cust, datetime.datetime.now()])
            writer.writerow(['Item', 'Description', 'Loc', 'Bay', 'Row', 'Col', 'Min', 'Max'])

            for b in range(self.bays):
                for r in range(self.bay_rows):
                    for c in range(self.bay_cols):
                        item = self.slots[b][r][c]
                        writer.writerow([item.id, 
                                         item.desc,
                                         f'{str(b+1).zfill(2)}.{ROWS[r]}{str(c+1).zfill(2)}',
                                         str(b+1).zfill(2), 
                                         ROWS[r], 
                                         str(c+1).zfill(2), 
                                         item.min, 
                                         item.max])

              
            
    def populate(self, items):
        '''Load all the priority items in a predetermined order into the pickface

        '''
        splits = split(items, self.bays)

        for b in range(self.bays):
            r = 0
            c = 0

            for index, row in splits[b].iterrows():
                item_info = items.loc[index]
                item = Item()
                item.set_id(index)
                item.set_desc(item_info.loc['description'])
                item.set_case_qty(item_info.loc['case_qty'])
                item.set_dimensions(item_info.loc['width'],
                                    item_info.loc['length'],
                                    item_info.loc['height'])
                item.set_minmax(item_info.loc['min'],
                                item_info.loc['max'])
                self.slots[b][self.row_priority[r]][self.col_priority[c]] = item
                c += 1 

                if c % len(self.col_priority) == 0:
                    c %= len(self.col_priority)
                    r += 1

                    if r % len(self.row_priority) == 0:
                        r %= len(self.row_priority)


    def load(self):
        '''Create empty items for every slot in the pickface
        
        '''
        for b in range(self.bays):
            rows = []
            for r in range(self.bay_rows):
                cols = []
                for c in range(self.bay_cols):
                    cols.append(Item())

                rows.append(cols)

            self.slots.append(rows)
    

    def get_item_by_id(self, id):
        '''Find an item within the pickface by id

        '''
        i = str(id)

        for b in range(self.bays):
            for r in range(self.bay_rows):
                for c in range(self.bay_cols):
                    if i == slots[b][r][c].id:
                        return {'slot': f'{str(b+1).zfill(2)}.{ROWS[r]}{str(c+1).zfill(2)}',
                                'info': slots[b][r][c].get_info()}

        return -1


    def get_item_by_slot(self, slot):
        '''Get item by slot

        '''j
        bay, row_col = slot.split('.')
        b = int(bay)
        r = ROWS.index(row)
        c = int(col)

        try:
            return slots[b][r][c].get_info()

        except:
            raise Exception(f'Invalid Slot: {slot}')     
    

class PF_9(Pickface):
    def __init__(self, cust, depth, height):
        self.cust = cust
        self.bays = 1
        self.bay_cols = 3
        self.bay_rows = 3
        self.depth = depth
        self.height = height
        self.slots = []
        self.col_priority = [1, 0, 2]
        self.row_priority = [1, 0, 2]

        self.load()


class PF_27(Pickface):
    def __init__(self, cust, depth, height):
        self.cust = cust
        self.bays = 3
        self.bay_cols = 3
        self.bay_rows = 3
        self.depth = depth
        self.height = height
        self.slots = []
        self.col_priority = [1, 0, 2]
        self.row_priority = [1, 0, 2]

        self.load()


class PF_32(Pickface):
    def __init__(self, cust, depth, height):
        self.cust = cust
        self.bays = 4
        self.bay_cols = 4
        self.bay_rows = 2
        self.depth = depth
        self.height = height
        self.slots = []
        self.col_priority = [2, 1, 0, 3]
        self.row_priority = [1, 0]

        self.load()


class PF_48(Pickface):
    def __init__(self, cust, depth, height):
        self.cust = cust
        self.bays = 4
        self.bay_cols = 4
        self.bay_rows = 3
        self.depth = depth
        self.height = height
        self.slots = []
        self.col_priority = [2, 1, 0, 3]
        self.row_priority = [1, 0, 2]
        
        self.load()


class Omni(Pickface):
    def __init__(self, cust, depth, height):
        self.cust = cust
        self.bays = 20
        self.bay_cols = 6
        self.bay_rows = 5
        self.depth = depth
        self.height = height
        self.slots = []
        
        self.load()
            
           
def split(items, num_bays):
    '''Find a good distribution of items accross the bays for the pickface

    '''
    items['visited'] = False
    t = []
    tmp = items.copy()
    num = int(len(items) / num_bays)

    # If one item alone is a higher percentage than can ideally fit into one bay,
    # take it and the lowest other items and make a bay.
    while True:
        t.append([])
        sum = items['percent'].iloc[0]
        tup = (items.index[0], items['percent'].iloc[0])
        t[-1].append(tup)
        items.iat[0, 10] = True

        for i in range(-1, -num, -1):
            t[-1].append((items.index[i], items['percent'].iloc[i]))
            sum += items['percent'].iloc[i]
            items.iat[i, 10] = True

        if sum >= (1 / num_bays) and num_bays != 1:
            items = items[items.visited == False]
            print(f'sum: {sum}')
            
        else:
            t.pop()
            break

    num_bays -= len(t)

    parts = [[] for p in range(num_bays)]

    i = 0
    #print(items[['orders', 'percent']])
    for index, row in items.iterrows():
        tup = (index, row['percent'])
        #print(tup)
        parts[i].append(tup)
        i = (i + 1) % num_bays
    
    sums = []

    for p in parts:
        sum = 0
        for i in p:
            sum += i[1]

        sums.append(sum)
    #print(sums)

    # Get the percent distribution of all the bays within 2% of each other
    max_s = max(sums)
    min_s = min(sums)

    # If the difference is greater than 2%, start swapping items
    while (max_s - min_s) > .02:
        max_ind = sums.index(max_s)
        min_ind = sums.index(min_s)
        
        swapped = False

        # Loop through the items of the least and greatest bay
        for i in range(len(parts[min_ind])-1, -1, -1):
            for j in range(len(parts[max_ind])-1, -1, -1):

                # If there's a larger item, put it into the smaller pickface
                if parts[max_ind][j][1] > parts[min_ind][i][1]:
                    tmp1, tmp2 = parts[min_ind][i], parts[max_ind][j]
                    parts[min_ind][i], parts[max_ind][j] = tmp2, tmp1
                    parts[max_ind].sort(key=lambda tup: tup[1], reverse=True)
                    parts[min_ind].sort(key=lambda tup: tup[1], reverse=True)
                    swapped = True
                    break
            
            if swapped:
                break

        # re-sums the distribution to calculate the difference
        sums = []
        for p in parts:
            sum = 0
            for i in p:
                sum += i[1]

            sums.append(sum)

        #print(sums)

        max_s = max(sums)
        min_s = min(sums)

    # Put eskew bays into the total distribution
    if t:
        for i in t:
            sum = 0
            for l in i:
                sum += l[1]

            sums.append(sum)
            parts.append(i)

    # Print distribution to the console
    if num_bays > 1:
        print(f'\nBay Load Distribution:')
        prefix = ''

        for s in range(len(sums)):
            print('{0}#{1}: {2:.2%}'.format(prefix, s + 1, sums[s]), end = '')
            prefix = ', '

        print('')

    for i in range(len(parts)):
        parts[i] = pd.DataFrame(parts[i], 
                                columns=['item_id', 'percent'])\
                                    .set_index('item_id')

    return parts