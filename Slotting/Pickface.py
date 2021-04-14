import pandas as pd
import math
from Item import *
import csv

ROWS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']


class Slot():
    def __init__(self, item, row, col, min, max):
        self.row = row 
        self.col = col
        self.item = item
        self.min = min
        self.max = max


class Pickface():

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


    #################################################
    # Print the PF to console
    #################################################
    def display(self):
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


    #################################################
    # Return all the items from the pickface
    #################################################
    def list_items(self):
        items = []

        for b in range(self.bays):
            for r in range(self.bay_rows):
                for c in range(self.bay_cols):
                    items.append(self.slots[b][r][c].id)

        return items


    #################################################
    # write the pickface to an excel file
    #################################################
    def to_csv(self):
        with open(f'data/{self.cust}-{self.bays * self.bay_cols * self.bay_rows}_slots.xlsx', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|')
            writer.writerow(['Customer:', self.cust])
            writer.writerow(['', 'Column'])
            cols = []
            
            for i in range(1, self.bay_cols + 1):
                cols.append(i)

            writer.writerow(['Row'] + cols * self.bays)
            
            for r in range(self.bay_rows - 1, -1, -1):
                row = [ROWS[r]]

                for b in range(self.bays):
                    for c in range(self.bay_cols):
                        row.append(self.slots[b][r][c].id)

                writer.writerow(row)
              
                
    #################################################
    # Load all the priority items in a predetermined order into the pickface
    #################################################
    def populate(self, items):
        #print(items[['orders', 'percent']])
        splits = split(items, self.bays)

        for b in range(self.bays):
            r = 0
            c = 0

            for index, row in splits[b].iterrows():
                item = Item()
                item.id = index
                self.slots[b][self.row_priority[r]][self.col_priority[c]] = item
                c += 1 

                if c % len(self.col_priority) == 0:
                    c %= len(self.col_priority)
                    r += 1

                    if r % len(self.row_priority) == 0:
                        r %= len(self.row_priority)


    #################################################
    # Create empty items for every slot in the pickface
    #################################################
    def load(self):
        for b in range(self.bays):
            rows = []
            for r in range(self.bay_rows):
                cols = []
                for c in range(self.bay_cols):
                    cols.append(Item())

                rows.append(cols)

            self.slots.append(rows)
    

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
            

#################################################
# Find a good distribution of items accross the bays for the pickface
#################################################           
def split(items, num_bays):

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
        items.iat[0, 3] = True

        for i in range(-1, -num, -1):
            t[-1].append((items.index[i], items['percent'].iloc[i]))
            sum += items['percent'].iloc[i]
            items.iat[i, 3] = True

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