import pandas as pd
pd.set_option('max_columns', None)
import math
from Item import *
import csv
import datetime as dt
import os
import numpy as np

ROWS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']


class Pickface():
    '''Pickface class that stores info about the pickface and stores an array of items.

    '''
    edit_date: dt.datetime
    bays: int 
    bay_cols: int
    bay_rows: int
    row_height: list
    depth: int
    slots: list 
    cust: str 
    row_priority: list
    col_priority: list


    def __init__(self, bays: int = 0, bay_cols: int = 0, bay_rows: int = 0, 
                 row_height: list = [], depth: int = 0, slots: list = [], 
                 cust: str = '', row_priority: list = [], 
                 col_priority: list = []):
        self.bays = bays
        self.bay_cols = bay_cols
        self.bay_rows = bay_rows
        self.row_height = row_height
        self.depth = depth
        self.slots = slots
        self.cust = cust
        self.row_priority = row_priority
        self.col_priority = col_priority



    def display(self):
        '''Print the PF to console

        '''
        print('')
        print(f'Customer: {self.cust}')
        print('###############################################')
        nope = ''
        for b in range(self.bays):
            row_del = ''
            for r in range(self.bay_rows - 1, -1, -1):
                print(row_del) 
                for c in range(self.bay_cols):
                    if self.slots[b][r][c] is not None:
                        print(f'| {str(b+1).zfill(2)}.{ROWS[r]}{str(c+1).zfill(2)}: {self.slots[b][r][c].id.rjust(10)} |', end='')
                    else:
                        print(f'| {str(b+1).zfill(2)}.{ROWS[r]}{str(c+1).zfill(2)}: {nope.rjust(10)} |', end='')

                row_del = '\n-----------------------------------------------'

            print('\n###############################################')



    def list_items(self):
        '''Return all the items from the pickface

        '''
        items = []

        for b in range(self.bays):
            for r in range(self.bay_rows):
                for c in range(self.bay_cols):
                    if self.slots[b][r][c] is not None:
                        items.append(self.slots[b][r][c].id)

        return items



    def to_csv(self, dir = 'data/'):
        '''Write the pickface to a csv file.

        '''
        filepath = os.path.join(dir, f'{self.cust}-{self.bays * self.bay_cols * self.bay_rows}.csv')
        print(f'Saving pickface to {filepath}... ', end = '')

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|')
            writer.writerow(['Date', 'Customer', 'Bays', 'Columns', 
                             'Col Priority', 'Rows', 'Row Height', 
                             'Row Priority', 'Depth'])
            writer.writerow([dt.datetime.now(), self.cust, self.bays, 
                             self.bay_cols, ';'.join(map(str, [x + 1 for x in self.col_priority])), 
                             self.bay_rows, ';'.join(map(str, self.row_height)), 
                             ';'.join([ROWS[x] for x in self.row_priority]), self.depth])
            writer.writerow([])
            writer.writerow(['Item', 'Description', 'Location', 'Bay', 'Row', 
                             'Column', 'Min', 'Max', 'Case Qty', 'Width', 
                             'Length', 'Height'])

            for b in range(self.bays):
                for r in range(self.bay_rows):
                    for c in range(self.bay_cols):
                        item = self.slots[b][r][c]
                        #print(item)
                        if item is not None:
                            writer.writerow([item.id, 
                                             item.desc,
                                             f'{str(b+1).zfill(2)}.{ROWS[r]}{str(c+1).zfill(2)}',
                                             b + 1, 
                                             ROWS[r], 
                                             c + 1, 
                                             item.min, 
                                             item.max,
                                             item.case_qty,
                                             item.width,
                                             item.length,
                                             item.height])

        print('Done')
              


    def from_csv(self, filepath):
        '''Read a pickface from a csv file as written by Pickface.to_csv().

        '''
        print(f'Reading pickface from {filepath}... ', end = '')
    
        with open(filepath, 'r', newline = '') as f:
            reader = csv.reader(f)
            try:
                next(reader) #skip table header
                info = next(reader)

                cust = info[1]
                bays = info[2]
                cols = info[3]
                col_pr = info[4].split(';')
                rows = info[5]
                row_he = info[6].split(';')
                row_pr = info[7].split(';')
                depth = info[8]

                self.bays = int(bays)
                self.bay_cols = int(cols)
                self.col_priority = [x - 1 for x in map(int, col_pr)]
                self.bay_rows = int(rows)
                self.row_priority = [ROWS.index(x) for x in row_pr]
                self.row_height = map(float, row_he)
                self.depth = int(depth)
                self.cust = cust.upper()

            except:
                print("ERROR: Pickface couldn't be loaded.")
                return 

            self.load()

            #skip the empty line and other table header
            next(reader)
            next(reader)

            for r in reader:

                id = str(r[0])
                desc = str(r[1])
                bay = int(r[3]) - 1
                row = ROWS.index(r[4])
                col = int(r[5]) - 1
                min = int(r[6])
                max = int(r[7])
                case_qty = int(float(r[8]))
                width = float(r[9])
                length = float(r[10])
                height = float(r[11])

                try:
                    item = Item(id = id, description = desc, height = height, 
                            width = width, length = length, min = min, 
                            max = max, case_qty = case_qty)
                    
                except:
                    print(f'\nInvalid item read:\n{r}')
                    continue

                #item.display()
                self.slots[bay][row][col] = item
                #print(self.slots)
                 
        print('Done')
        #self.display()



    def populate(self, items: pd.DataFrame):
        '''Load all the priority items in a predetermined order into the pickface

        '''
        splits = split(items, self.bays)

        for b in range(self.bays):
            r = 0
            c = 0

            for index, row in splits[b].iterrows():
                item_info = items.loc[index].to_dict()
                slotted = False

                item = Item(id = index,
                            description = item_info['description'],
                            case_qty = item_info['case_qty'],
                            width = item_info['width'],
                            length = item_info['length'],
                            height = item_info['height'],
                            min = item_info['min'],
                            max = item_info['max'])

                for row in self.row_priority:
                    if item.height is None or\
                        self.row_height[row] >= item.height \
                        or math.isnan(item.height):

                        for col in self.col_priority:
                            if self.slots[b][row][col] is None:
                                self.slots[b][row][col] = item
                                slotted = True
                                break
                    if slotted:
                        break

                if not slotted:
                    print(f'Item {item.id} could not be slotted.')

           

    def load(self):
        '''Create empty splaces for every slot in the pickface.
        
        '''
        for b in range(self.bays):
            rows = []
            for r in range(self.bay_rows):
                cols = []
                for c in range(self.bay_cols):
                    cols.append(None)

                rows.append(cols)

            self.slots.append(rows)
    

    def get_item_by_id(self, id):
        '''Find an item within the pickface by its id.

        '''
        i = str(id)

        for b in range(self.bays):
            for r in range(self.bay_rows):
                for c in range(self.bay_cols):
                    if i == slots[b][r][c].id:
                        return slots[b][r][c].get_info()

        return -1


    def get_item_by_slot(self, slot):
        '''Get item by slot, such as "02.B04 "

        '''
        bay, row_col = slot.split('.')
        row = row_col[:1]
        col = row_col[1:]
        
        b = int(bay)
        r = ROWS.index(row)
        c = int(col)

        try:
            return slots[b][r][c].get_info()

        except:
            raise Exception(f'Invalid Slot: {slot}')   
        
    def evaluate(self, hashkey):
        '''Evaluate a pickface using a hashkey of orders.

        '''
        print('\nEvaluating pickface')
        items = self.list_items()

        order_count = hashkey.order_config.value_counts().to_frame()\
            .rename(columns={'order_config': 'order_count'})
        order_count['visited'] = False
        #print(order_count)
        ord_serv = 0
        ord_sum = order_count.order_count.sum()

        for index, row in order_count.iterrows():
            if all(x in items for x in index.split(';')):
                order_count.at[index, 'visited'] = True
                ord_serv += row['order_count']
                #print(index)

        print('\nTotal Orders: {0:,}'.format(ord_sum))
        print('Orders Served by PF: {0:,}'.format(ord_serv))
        ord_per = ord_serv / ord_sum
        print('% Orders Served: {0:.2%}'.format(ord_per))

        visited = list(order_count[order_count.visited == True].index)
        sub_val_count = sub_hashkey['date'].value_counts()

        min = int(round(sub_val_count.min()))
        q1 = int(round(np.nanpercentile(sub_val_count, 25)))
        med = int(round(sub_val_count.median()))
        q3 = int(round(np.nanpercentile(sub_val_count, 75)))
        max = int(round(sub_val_count.max()))

        print('\tOrders/Day:')
        print(f'\tMin = {min:,}')
        print(f'\t1Qt = {q1:,}')
        print(f'\tMed = {med:,}')
        print(f'\t3Qt = {q3:,}')
        print(f'\tMax = {max:,}')

    

class PF_9(Pickface):
    def __init__(self, cust, depth, row_height):
        self.cust = cust
        self.bays = 1
        self.bay_cols = 3
        self.bay_rows = 3
        self.depth = depth
        self.row_height = row_height
        self.slots = []
        self.col_priority = [1, 0, 2]
        self.row_priority = [1, 0, 2]

        self.load()


class PF_27(Pickface):
    def __init__(self, cust, depth, row_height):
        self.cust = cust
        self.bays = 3
        self.bay_cols = 3
        self.bay_rows = 3
        self.depth = depth
        self.row_height = row_height
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
        self.row_height = height
        self.slots = []
        self.col_priority = [2, 1, 0, 3]
        self.row_priority = [1, 0]

        self.load()


class PF_48(Pickface):
    def __init__(self, cust, depth, row_height):
        self.cust = cust
        self.bays = 4
        self.bay_cols = 4
        self.bay_rows = 3
        self.depth = depth
        self.row_height = row_height
        self.slots = []
        self.col_priority = [2, 1, 0, 3]
        self.row_priority = [1, 0, 2]
        
        self.load()


class Omni(Pickface):
    def __init__(self, cust, depth, row_height):
        self.cust = cust
        self.bays = 20
        self.bay_cols = 6
        self.bay_rows = 5
        self.depth = depth
        self.row_height = row_height
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