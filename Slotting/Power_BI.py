import pandas as pd
pd.set_option('max_columns', None)
#pd.set_option('max_rows', None)
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


import math
import csv
import datetime as dt
import os
import copy
import threading
import matplotlib.pyplot as plt

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


    def __init__(self, bays: int = 0, bay_rows: int = 0, bay_cols: int = 0, 
                 row_height: list = [0], depth: int = 1, slots: list = [], 
                 cust: str = ''):
        self.bays = bays
        self.bay_cols = bay_cols
        self.bay_rows = bay_rows

        if not row_height:
            for r in bay_rows:
                self.row_height.append(99)
        
        else:
            self.row_height = row_height

        self.depth = depth
        self.slots = copy.deepcopy(slots)
        self.cust = cust
        self.row_priority = []
        self.col_priority = []

        for r in range(self.bay_rows - 1, -1 ,-1):
            self.row_priority.append(r)
        self.row_priority.append(bay_rows - 1)

        dec = int(math.ceil(bay_cols / 2)) - 1
        inc = dec + 1
        dir = -1

        while len(self.col_priority) < bay_cols:
            if dir > 0:
                self.col_priority.append(inc)
                inc += 1

            else:
                self.col_priority.append(dec)
                dec -= 1

            dir *= -1
        
        self.load()



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



    def to_csv(self, dir = r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\Slotting\Slotting\data\\"):
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

                id = str(r[0]).strip()
                desc = str(r[1]).strip()
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
        #print(hex(id(self.slots)))
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
                    if item.height is None or self.row_height[row] >= item.height or math.isnan(item.height):

                        for col in self.col_priority:
                            if self.slots[b][row][col] is None:
                                self.slots[b][row][col] = item
                                #self.display()
                                slotted = True
                                break
                    if slotted:
                        break
                    

                if not slotted:
                    print(f'Item {item.id} could not be slotted: {item.height}, {self.row_height}')
                    #self.display()

           

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
        sub_hashkey = hashkey[hashkey.order_config.isin(visited)]
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

    # Get the percent distribution of all the bays within 2% of each other
    max_sum = max(sums)
    min_sum = min(sums)

    # If the difference is greater than 2%, start swapping items
    while (max_sum - min_sum) > .02:
        max_ind = sums.index(max_sum)
        min_ind = sums.index(min_sum)
        
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

        max_sum = max(sums)
        min_sum = min(sums)

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


import pyodbc

def connect_db(DB: str = 'Items'):
    '''Connect to a Microsoft Access database localy stored.

    '''
    string = r'DBQ=C:\Users\David.Moreno\OneDrive - Visible SCM\db\{0:s}.accdb;'.format(DB)
    #print(string)
    try:
        connection = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};' + string)
        connection.autocommit = False
        return connection

    except pyodbc.Error as err:
        raise Exception(err)
        return -1
    
    #cursor = conn.cursor()
    #cursor.execute('select * from Customer')
   
    #for row in cursor.fetchall():
    #    print (row)



def load_hashkey(filepath):
    '''Load a Power BI Hashkey report from a file and generate order configurations

    '''
    print(f'\nLoading hashkey: {filepath} . . . ', end = '')

    df = None

    # Get the file type and read it in appropriately
    f_type = filepath.split('.')[-1]
    if f_type == 'csv':
        df = pd.read_csv(filepath,
                         dtype = 'string')

    elif f_type == 'xlsx':
        df = pd.read_excel(filepath,
                           dtype = 'string')

    else:
        raise f"Unrecognized filetype: .{f_type}"

    if 'Optimization Hash Key' in df.columns:
        df = df.rename(columns = {'Created Date': 'date',
                             'Optimization Hash Key': 'hashkey',
                             'Client Order Number': 'order_number'})
    df = df.set_index('order_number')

    df['date'] = pd.to_datetime(df['date']).dt.date
    
    recent = df['date'].max()
    d = recent - dt.timedelta(days=42)
    df = df[df['date'] >= d]
    

    # Create order configuration if not already in dataframe
    if 'order_config' not in df.columns:
        df['order_config'] = ''
        df['hashkey'] = df['hashkey'].str.replace(r';$', '')

        # Order configurations are hashkeys without quantities
        print('\nGenerating Order Configurations . . . ', end = '')
        for ind, row in df.iterrows():
            hashes = row['hashkey'].split(';')
            config = ''
            prefix = ''
            for hash in hashes:
                if hash != '':
                    config += str(prefix + hash.split('*')[0])
                    prefix = ';'

            df.at[ind, 'order_config'] = config

        if f_type == 'csv':
            df.to_csv(filepath)

        elif f_type == 'xlsx':
            df.to_excel(filepath) 

    

    print('Done')

    return df



def generate_hashkey(df):
    '''Take Order numbers, items, and quantities to build a hash key. 
    Use this if each item from an order is on its own line

    '''
    print('\nGenerating hashkey from dataframe...', end = '')
    # Join each item and its quantity into a string on the same row
    df['hashkey'] = df.apply(lambda row: row.item_id + '*' + str(row.quantity), 
                             axis = 1)
    
    # Join all rows of an order into a hashkey string
    df = df.groupby(['order_number'])['hashkey']\
        .apply(lambda row: ';'.join(row)).reset_index()
    print('Done')
    return df



def generate_hashkey_ASC(filepath, cust):
    '''Take a file from ASC and generate hashkeys and order configs

    '''
    print(f'\nGenerating Hashkey from ASC file: {filepath} . . . ')

    df = None

    # Get the file type and read it in appropriately
    f_type = filepath.split('.')[-1]
    if f_type == 'csv':
        df = pd.read_csv(filepath,
                         dtype = 'string')

    elif f_type == 'xlsx':
        df = pd.read_excel(filepath,
                           dtype = 'string')

    print(df)

    dict = {}
    hash = []
    config = []  
    n_frame = []    # Stores all items in the desired format
    n_row = []      # Keeps each order together in one row

    print('Loading kit and item info . . . ', end = '')

    cnxn = connect_db()
    if type(cnxn) is int:
        return

    crsr = cnxn.cursor()

    item_sql = '''SELECT i.ASC_id 
                  FROM Item AS i
                  WHERE i.customer = ucase('{0:s}');'''.format(cust)

    items = pd.read_sql(item_sql, cnxn).set_index('ASC_id')
    item_id = list(items.index.values)
    
    
    kit_sql = '''SELECT k.ASC_id
                 FROM Kit AS k
                 WHERE k.customer = ucase('{0:s}');'''.format(cust)

    #kits = pd.read_sql(kit_sql, cnxn).set_index('kit_id')

    #kit_id = list(kits.index.values)
    
    crsr.close()
    cnxn.close()

    print('Done\nBuilding hashkey from items and kits . . . ', end = '')

    for index, row in df.iterrows():
        
        # If the it item can be converted into a date, it's a new item
        try:
            date_time = dt.datetime.strptime(row['SHIPDATE'], 
                                                   "%m/%d/%Y %I:%M:%S %p")
            for i in sorted(dict.keys()):
                hash.append(str(i + '*' + dict[i]))
                config.append(i)

            n_row.append(';'.join(hash))
            n_row.append(';'.join(config))
            n_frame.append(n_row.copy())
            dict.clear()
            hash.clear()
            config.clear()
            
            #print(n_frame)
            n_row.clear()
            n_row.append(date_time)
            n_row.append(row['ORDERNUMBER'])

        except:
            if row['ORDERNUMBER'] in item_id:

                if row['ORDERNUMBER'] in dict.keys():
                    dict[row['ORDERNUMBER']] += row['SHIPDATE']

                else:
                    dict[row['ORDERNUMBER']] = row['SHIPDATE']

            #elif row['ORDERNUMBER'] in kit_id:
            #    continue


    for i in sorted(dict.keys()):
        hash.append(str(i + '*' + dict[i]))
        config.append(i)

    n_row.append(';'.join(hash))
    n_row.append(';'.join(config))
    n_frame.append(n_row.copy())

    # first item is a blank line, so pop it
    n_frame.pop(0)
    hashkey = pd.DataFrame(n_frame, columns = ['date', 'order_number', 
                                               'hashkey', 'order_config'])

    hashkey = hashkey[hashkey['hashkey'] != '']
    hashkey = hashkey.set_index('order_number')
    hashkey.to_csv(f'data/{cust}_hashkey.csv')
    print('Done')
    return hashkey



def generate_hashkey_from_SQL(filepath):
    '''Take an exported file from the SQL server and generate hashkeys and 
    order configs
    
    WARNING: SQL server puts 0's when an order shipped incompletely instead of 
    displaying the number ordered

    '''
    df = None

    # Get the file type and read it in appropriately
    f_type = filepath.split('.')[-1]
    if f_type == 'csv':
        df = pd.read_csv(filepath,
                         dtype = 'string',
                         header = 1)\
                               .rename(columns = {'Ship Date': 'date',
                                                  'Order Number': 'order_number',
                                                  'Item ID': 'item_id',
                                                  'Qty\nShipped': 'qty'})

    elif f_type == 'xlsx':
        df = pd.read_excel(filepath,
                           dtype = 'string',
                           header = 1)\
                               .rename(columns = {'Ship Date': 'date',
                                                  'Order Number': 'order_number',
                                                  'Item ID': 'item_id',
                                                  'Qty\nShipped': 'qty'})

    #print(df.head())

    # Optional: substitute the 0's for 1 or another number
    #df['qty'] = df.apply(lambda row: '1' if row.qty == '0' else row.qty, axis = 1)

    # We don't need all datetime so convert to date
    df['date'] = pd.to_datetime(df['date']).dt.date
  
    df['hashkey'] = df.apply(lambda row: row.item_id + '*' + row.qty, axis = 1)
    
    hashkey = df.groupby(['order_number', 'date'])['hashkey'].apply(lambda row: ';'.join(row)).reset_index()

    print(hashkey.head())
    return hashkey



def min_max_from_hashkey(hashkey, case_info):
    '''Calculate a min-max from a hashkey using 80th and 90th 
    percentiles of items shipped each day as min and max
    and using master case quantities.

    '''
    print('\nCalculating Min-Max from hashkey . . . ', end = '')

    # Split hashkey by each item while keeping the date
    df = pd.concat([pd.Series(row['date'], row['hashkey'].split(';')) for _, row in hashkey.iterrows()]).reset_index()
    df = df.rename(columns = {'index': 'hashkey', 0: 'date'})
    df = df[df['hashkey'] != '']
    print(df)
    
    # Split the hashkey into items and quantities
    df[['item_id', 'qty']] = df['hashkey'].str.split('*', expand=True)
    try:
        df['qty'] = df['qty'].astype(float)
    except:
        print(df['qty'].max())
    #print(df)

    # Get the total shipped each day
    df = df.groupby(['date', 'item_id']).agg({'qty': 'sum'})
    df.reset_index(inplace=True)
    print(df)

    # Pivot so that day sums of all item quantities are aligned on the same row with the date
    pivot_df = df.pivot(index = 'date', columns = 'item_id', values = 'qty')
    min_max = pd.DataFrame(columns = ['item_id', '80', '90'])
    pivot_df = pivot_df.fillna(0)
    print(pivot_df)

    for i in range(0, len(pivot_df.columns)):
        min_max = min_max.append({'item_id': str(pivot_df.columns[i]),
                                  '80': np.nanpercentile(pivot_df[pivot_df.columns[i]], 80), 
                                  '90': np.nanpercentile(pivot_df[pivot_df.columns[i]], 90)}, 
                                 ignore_index = True)

    #print(min_max)
    case_info = case_info[['case_qty']]
    min_max = min_max.set_index('item_id').join(case_info, how='left')

    #print(min_max)
    min_max['min'] = min_max.apply(lambda row: to_int(np.ceil(row['80'] / row['case_qty'])) if row['case_qty'] is not None else 0, axis = 1)
    min_max['max'] = min_max.apply(lambda row: to_int(np.ceil(row['90'] / row['case_qty'])) if row['case_qty'] is not None else 0, axis = 1)
    
    print('Done')
    print(min_max)

    return min_max[['min', 'max']]



def remove_lol(hashkey: pd.DataFrame):
    '''Filters out any orders in the hashkey that could be handled in LOL

    '''
    print("\nRemoving LOL orders . . . ", end = "")
    tot_ord = len(hashkey)
    by_date = hashkey.groupby('date')['hashkey'].value_counts().to_frame()
    #print(by_date)
    for ind, row in by_date.iterrows():
        if row['hashkey'] >= 50:
            hash = ind[1]
            sum = 0
            for h in hash.split(";"):
                sum += int(h.split("*")[-1])

            if sum <= 10:
                hashkey = hashkey.drop(hashkey[(hashkey['date'] == ind[0]) & (hashkey.hashkey == ind[1])].index)

    #print(hashkey.groupby('date')['hashkey'].value_counts().to_frame())
    print('Done')
    print('Total Orders: {0:,}'.format(tot_ord))
    print('Orders Removed: {0:,} ({1:.2%})'.format(tot_ord - len(hashkey), (tot_ord - len(hashkey)) / tot_ord))
    print('Remaining: {0:,} ({1:.2%})'.format(len(hashkey), len(hashkey) / tot_ord))

    return hashkey



def to_int(val):
    '''Format NaN's, None's, and others into integers
    
    '''
    if math.isnan(val) or val is None or pd.isna(val):
        return ''
    else:
        return int(val)



import numpy as np


def overlaying_slotting(hashkey, pf, cust, row_height, **kwargs):
    '''Builds the desired pickfaces based on the most popular order configurations.
    It is "overlayed" because no items will repeat across any of the pickfaces, incomplete orders get passed to the next pickface.

    '''
    # convert the type to an iterable list
    if type(pf) is not list:
        pf = [pf]

    # Convert each item in pickface list to integer
    for i in range(len(pf)):
        try:
            pf[i] = int(pf[i])
        except:
            raise TypeError(f'Pickface number {i + 1} was not valid.')
    
    # Connect to DB
    cnxn = connect_db()
    if type(cnxn) is int:
        return

    cust_sql = '''SELECT customer_id
                  FROM Customer;'''

    cust_df = pd.read_sql(cust_sql, cnxn)

    custs = list(cust_df['customer_id'])

    if cust not in custs:
        raise Exception(f'"cust" must be an ASC Customer ID.\n"{cust}" not recognized!')
    
    # Get ignored or required items
    ignored = kwargs.get('ignore')
    required = kwargs.get('require')

    #print(hashkey)
    order_count = hashkey.order_config.value_counts().to_frame()\
        .rename(columns={'order_config': 'order_count'})

    order_count['visited'] = False
    print(order_count)

    # Connect to DB to get item info

    item_sql = '''SELECT i.item_id, i.description, i.case_qty, i.width, i.length, i.height
                  FROM Item AS i
                  WHERE i.customer = ucase('{0:s}');'''.format(cust)

    item_info = pd.read_sql(item_sql, cnxn).set_index('item_id')

    top = []        # Store top X items for each pickface
    pickfaces = []  # Store each pickface
    # Order sums for statistics reporting
    ord_sum = order_count.order_count.sum()
    backup = []     # Incomplete order configurations to use for later

    if ignored:
        print('Removing orders with ignored items . . . ', end = '')
        for ind, row in order_count.iterrows():
            # If any items in the order_count index match, delete the configuration
            if any(x in ignored for x in ind.split(';')):
                order_count = order_count.drop(ind)

        print('Done')

    # loop through each pickface number and get those top X items
    for p in range(len(pf)):
        configs = 0 # Number for tracking order configurations
        # each pickface is stored in its own dataframe
        top.append(pd.DataFrame(columns = ['item_id', 'orders', 'order_configs']))
        backup = []

        if required:
            for req in required:
                tmp = pd.DataFrame([[req, 0, []]], 
                                   columns = ['item_id', 'orders', 'order_configs'])
                top[p] = top[p].append(tmp, ignore_index = True)

        for ind, row in order_count.iterrows():
            items = ind.split(';')

            
            # if there aren't enough spaces to hold the next order configuration...
            # count the number of items missing and if they can fit in, add the config
            # and items
            if len(items) > (pf[p] - len(top[p])):
                not_in = 0
                for i in items:
                    if i not in list(top[p]['item_id']):
                        not_in += 1

                if not_in > (pf[p] - len(top[p])):
                    for i in items:
                        backup.append([i, row['order_count']])

                    continue

            # track if order configuration has been accounted for
            order_count.at[ind, 'visited'] = True
            h2 = []

            for i in items:

                # add the count if the item is already in the pickface
                if i in list(top[p]['item_id']):
                    #print(top[p])
                    ind_i = top[p][top[p]['item_id'] == i].index.values[0]
                    tmp = top[p].at[ind_i, 'orders']

                    top[p].at[ind_i, 'orders'] = tmp + row['order_count']
                    top[p].at[ind_i, 'order_configs'].append((str(configs), row['order_count']))

                else:
                    tmp = pd.DataFrame([[i, row['order_count'], [(str(configs), row['order_count'])]]], 
                                        columns = ['item_id', 'orders', 'order_configs'])
                    top[p] = top[p].append(tmp, ignore_index = True)
                    
            configs += 1

        print('')
        while len(top[p].index) < pf[p]:
            
            if (backup[0][0] not in list(top[p].item_id)):
                print(f'incomplete order configuration: {backup[0][0]}')
                tmp = pd.DataFrame([backup[0].append((str(configs), 0))], columns = ['item_id', 'orders', 'order_configs'])
                top[p] = top[p].append(tmp, ignore_index = True)
            backup.pop(0)

            if len(backup) == 0:
                break


        # Calculate the percent of total orders served from the pickface
        sum = top[p].orders.sum()
        top[p]['percent'] = top[p].apply(lambda row: row.orders / sum, axis = 1)

        top[p] = top[p].sort_values('orders', ascending = False).set_index('item_id')
        print(f'\nTop {len(top[p])} Items:\n{top[p]}')
        top[p] = top[p].join(item_info, how = "left")
        #print(top[p])
        visited = list(order_count[order_count.visited == True].index)
        #print(visited)

        ord_serv = order_count[order_count.visited == True].order_count.sum()
        print('\nTotal Orders: {0:,}'.format(ord_sum))
        print('Ideal Conditions:')
        ord_per = ord_serv / ord_sum
        print('\t% Orders Served: {0:.2%}'.format(ord_per))

        sub_hashkey = hashkey[hashkey.order_config.isin(visited)]
        sub_val_count = sub_hashkey['date'].value_counts()

        d_min = int(round(sub_val_count.min()))
        q1 = int(round(np.nanpercentile(sub_val_count, 25)))
        d_med = int(round(sub_val_count.median()))
        q3 = int(round(np.nanpercentile(sub_val_count, 75)))
        d_max = int(round(sub_val_count.max()))

        print('\tOrders/Day:')
        print(f'\tMin = {d_min:,}')
        print(f'\t1Qt = {q1:,}')
        print(f'\tMed = {d_med:,}')
        print(f'\t3Qt = {q3:,}')
        print(f'\tMax = {d_max:,}')

        min_max = min_max_from_hashkey(sub_hashkey, item_info)
        #print(min_max)

        top[p] = top[p].join(min_max, how = "left")

        # Remove all the used order configurations
        order_count = order_count[order_count.visited != True]

        pickf = switch_pf(pf[p], cust, 1, row_height)
        print(top[p])
        pickf.populate(top[p])
        #pickf.display()
        #pickf.evaluate(hashkey)
        pickf.to_csv()
        pickfaces.append(pickf)

    return pickfaces



def slotting(hashkey, pf, cust, heights, **kwargs):
    '''Builds the desired pickfaces based on the most popular order configurations.

    '''
    frame = []
    # convert the type to an iterable list
    if type(pf) is not list:
        raise TypeError(f'Pickface info was not valid.')

    # Convert each item in pickface list to integer
    for i in range(len(pf)):
        if pf[i][1] != len(heights[i]):
            raise "Pickface row number and given heights don't match."

    # Connect to DB
    cnxn = connect_db()
    if type(cnxn) is int:
        return

    cust_sql = '''SELECT customer_id
                  FROM Customer;'''

    cust_df = pd.read_sql(cust_sql, cnxn)

    custs = list(cust_df['customer_id'])

    if cust not in custs:
        raise Exception(f'"cust" must be an ASC Customer ID.\n"{cust}" not recognized!')
    
    # Get ignored or required items
    ignored = kwargs.get('ignore')
    required = kwargs.get('require')

    #print(hashkey)
    order_count = hashkey.order_config.value_counts().to_frame()\
        .rename(columns={'order_config': 'order_count'})

    order_count['visited'] = False
    print(order_count)
    print(len(order_count[order_count.order_count == 1]))

    # Connect to DB to get item info
    item_sql = '''SELECT i.ASC_id AS item_id, i.description, i.case_qty, i.width, i.length, i.height
                  FROM Item AS i
                  WHERE i.customer = ucase('{0:s}');'''.format(cust)

    item_info = pd.read_sql(item_sql, cnxn).set_index('item_id')

    #print(item_info)

    cnxn.close()

    top = []        # Store top X items for each pickface
    pickfaces = []  # Store each pickface
    # Order sums for statistics reporting
    ord_sum = order_count.order_count.sum()
    backup = []     # Incomplete order configurations to use for later

    if ignored:
        print('Removing orders with ignored items . . . ', end = '')
        for ind, row in order_count.iterrows():
            # If any items in the order_count index match, delete the configuration
            if any(x in ignored for x in ind.split(';')):
                order_count = order_count.drop(ind)

        print('Done')

    # loop through each pickface number and get those top X items
    for p in range(len(pf)):
        slots = pf[p][2] * pf[p][1] * pf[p][0]
        configs = 0 # Number for tracking order configurations
        # each pickface is stored in its own dataframe
        top.append(pd.DataFrame(columns = ['item_id', 'orders', 'order_configs']))
        backup = []

        if required:
            for req in required:
                tmp = pd.DataFrame([[req, 0, []]], 
                                   columns = ['item_id', 'orders', 'order_configs'])
                top[p] = top[p].append(tmp, ignore_index = True)

        for ind, row in order_count.iterrows():
            items = ind.split(';')

            
            # if there aren't enough spaces to hold the next order configuration...
            # count the number of items missing and if they can fit in, add the config
            # and items
            if len(items) > (slots - len(top[p])):
                not_in = 0
                for i in items:
                    if i not in list(top[p]['item_id']):
                        not_in += 1

                if not_in > (slots - len(top[p])):
                    for i in items:
                        backup.append([i, row['order_count']])

                    continue

            # track if order configuration has been accounted for
            order_count.at[ind, 'visited'] = True
            h2 = []

            for i in items:

                # add the count if the item is already in the pickface
                if i in list(top[p]['item_id']):
                    #print(top[p])
                    ind_i = top[p][top[p]['item_id'] == i].index.values[0]
                    tmp = top[p].at[ind_i, 'orders']

                    top[p].at[ind_i, 'orders'] = tmp + row['order_count']
                    top[p].at[ind_i, 'order_configs'].append((str(configs), row['order_count']))

                else:
                    tmp = pd.DataFrame([[i, row['order_count'], [(str(configs), row['order_count'])]]], 
                                        columns = ['item_id', 'orders', 'order_configs'])
                    top[p] = top[p].append(tmp, ignore_index = True)
                    
            configs += 1

        print('')
        while len(top[p].index) < slots:
            
            if (backup[0][0] not in list(top[p].item_id)):
                print(f'incomplete order configuration: {backup[0][0]}')
                tmp = pd.DataFrame([backup[0].append((str(configs), 0))], columns = ['item_id', 'orders', 'order_configs'])
                top[p] = top[p].append(tmp, ignore_index = True)
            backup.pop(0)

            if len(backup) == 0:
                break


        # Calculate the percent of total orders served from the pickface
        sum = top[p].orders.sum()
        top[p]['percent'] = top[p].apply(lambda row: row.orders / sum, axis = 1)

        top[p] = top[p].sort_values('orders', ascending = False).set_index('item_id')
        print(f'\nTop {len(top[p])} Items:\n{top[p]}')
        top[p] = top[p].join(item_info, how = "left")
        #print(top[p])
        visited = list(order_count[order_count.visited == True].index)
        #print(visited)
        
        ord_serv = order_count[order_count.visited == True].order_count.sum()
        print('\nTotal Orders: {0:,}'.format(ord_sum))
        print('Ideal Conditions:')
        ord_per = ord_serv / ord_sum
        print('\t% Orders Served: {0:.2%}'.format(ord_per))

        sub_hashkey = hashkey[hashkey.order_config.isin(visited)]
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

        min_max = min_max_from_hashkey(sub_hashkey, item_info)
        #print(min_max)

        top[p] = top[p].join(min_max, how = "left")

        # Remove all the used order configurations
        order_count = order_count[order_count.visited != True]

        pickf = Pickface(pf[p][0], pf[p][1], pf[p][2], heights[p], 1, cust = cust)
        #pickfaces.append(Pickface(pf[p][0], pf[p][1], pf[p][2], heights[p], 1, cust = cust, row_priority = prior[p]))
        #pickf = switch_pf(pf[p], cust, 1, row_height)
        #print(top[p])
        pickf.populate(top[p])
        #pickf.display()
        #pickf.evaluate(hashkey)
        #pickf.to_csv()
        pickfaces.append(copy.deepcopy(pickf))
        
        for b in range(pickf.bays):
            for r in range(pickf.bay_rows):
                for c in range(pickf.bay_cols):
                    item = pickf.slots[b][r][c]
                    #print(item)
                    if item is not None:
                        frame.append([p,
                                        f'{str(b+1).zfill(2)}.{ROWS[r]}{str(c+1).zfill(2)}',
                                        item.id, 
                                        item.desc,
                                        b + 1, 
                                        ROWS[r], 
                                        c + 1, 
                                        item.min, 
                                        item.max,
                                        item.case_qty,
                                        item.width,
                                        item.length,
                                        item.height])

    columns = ['pickface', 'location', 'item_id', 'desc', 'bay' ,'row', 'col', 'min', 'max', 'case_qty', 'width', 'length', 'height']

    
    return pd.DataFrame(frame, columns = columns)



def build_by_velocity(hashkey: pd.DataFrame, num: int):
    '''Build a pickface strictly by item velocity

    '''
    hashkey_count = hashkey.hashkey.value_counts().to_frame()
    print(hashkey_count)

    items = pd.DataFrame(columns = ['item_id', 'count'])\
        .set_index('item_id')
    #print(items)

    for index, row in hashkey_count.iterrows():
        hash = index.split(';')

        for h in hash:
            i:str = ''
            c:int = 0

            try:
                i, c = h.split('*')
            except:
                break
            c = int(c)

            if i in items.index:
                tmp = items.at[i, 'count']

                items.at[i, 'count'] = tmp + (c * row['hashkey'])

            else:
                tmp = pd.DataFrame([[i, c * row['hashkey']]], 
                                   columns = ['item_id', 'count'])\
                                       .set_index('item_id')
                #print(tmp)
                items = items.append(tmp)

            #print(items)

    print(items)
    items = items.sort_values('count', ascending = False)
    sum = items['count'].sum()
    #items['percent'] = items.apply(lambda row: row['count'] / sum, axis = 1)
    #print(len(items[items.percent >= 0.005]))
    items = items[:num]
    for ind, row in items.iterrows():
        print(row)

    top = items.index
    order_count = hashkey.order_config.value_counts().to_frame()\
            .rename(columns={'order_config': 'order_count'})
    order_count['visited'] = False
    #print(order_count)
    ord_serv = 0
    ord_sum = order_count.order_count.sum()

    for index, row in order_count.iterrows():
        if all(x in top for x in index.split(';')):
            order_count.at[index, 'visited'] = True
            ord_serv += row['order_count']
            #print(index)

    print('\nTotal Orders: {0:,}'.format(ord_sum))
    print('Orders Served by PF: {0:,}'.format(ord_serv))
    ord_per = ord_serv / ord_sum
    print('% Orders Served: {0:.2%}'.format(ord_per))

    visited = list(order_count[order_count.visited == True].index)
    sub_hashkey = hashkey[hashkey.order_config.isin(visited)]
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

    return items.index



def single_single(filepath, **kwargs):
    '''Find statistics on the single-single orders in a hashkey.

    '''
    print(f'\nUploading batch orders to DB: {filepath} . . . ')

    df = None

    # Get the file type and read it in appropriately
    f_type = filepath.split('.')[-1]
    if f_type == 'csv':
        df = pd.read_csv(filepath,
                         dtype = 'string')

    elif f_type == 'xlsx':
        df = pd.read_excel(filepath,
                           dtype = 'string')

    print(df)

    order = ''

    print('Loading item info . . . ', end = '')

    cnxn = connect_db('single-single')
    if type(cnxn) is int:
        return

    crsr = cnxn.cursor()

    item_sql = '''SELECT ITEMID
                  FROM Item;'''

    crsr.execute(item_sql)

    item_id = []
    for row in crsr.fetchall():
        item_id.append(row[0])
    
    bo = []
    oi = []

    #crsr.execute('''INSERT INTO Batch_Order (order_num, batch_num) VALUES('1', 1);''')
    #cnxn.commit()
    print('Done\nCompiling Batch Orders and Items  . . . ', end = '')

    for index, row in df.iterrows():
        # If the it item can be converted into a date, it's a new item
        try:
            date_time = dt.datetime.strptime(row['OURORDERDATE'], 
                                             "%m/%d/%Y %I:%M:%S %p")
            order = row['ORDERNUMBER']
            bo.append((row['OURORDERDATE'], order, row['BATCH_NUM']))

        except:
            if row['ORDERNUMBER'] in item_id:
                oi.append((order, row['ORDERNUMBER']))
 
    print('Done')
    print(f'# orders found: {len(bo)}')
    print(f'# items found: {len(oi)}')

    print('\nInserting Batch Orders and Items into DB . . . ', end = '')

    try:
        crsr.executemany("INSERT INTO Batch_Order (order_date, order_num, batch_num) VALUES(?, ?, ?)",
                         bo)
        cnxn.commit()
    except pyodbc.Error as err:
        print(err)

    try:
        crsr.executemany("INSERT INTO Order_Item (order_num, item_id) VALUES(?, ?)",
                         oi)
        cnxn.commit()
    except pyodbc.Error as err:
        print(err)
    
    
    print('Done')

    return



POD = [1, 3, 3]
HVPNP = [3, 3, 3]
PNP = [4, 3, 4]

MIN_LOL_ORDERS = 50
MAX_LOL_LINES = 3
MAX_LOL_ITEMS = 5


def pf_switch(items):
    if len(items) <= 9:
        return 'POD'

    elif len(items) <= 27:
        return 'HVPnP'

    else:
        return 'PnP'

def pf_dim_switch(pf: str):
    if pf == 'POD':
        return POD
    elif pf == 'HVPnP':
        return HVPNP
    elif pf == 'PnP':
        return PNP 

def slotting(info, frame, pf_info):

    hashkey = load_hashkey(info['hashkey'])

    lol = True if info['lol'] == 'True' else False

    pf = [pf_dim_switch(p) for p in info['pfs'].split(';')]

    h1 = info['heights'].split(';')
    heights = []
    for h2 in h1:
        heights.append([int(h) for h in h2.split(',')])

    cust = info['client']

    ignored = str(info['ignored']).split(';') if not pd.isna(info['ignored']) else None
    required = str(info['required']).split(';') if not pd.isna(info['required']) else None
    
    print(f'client: {cust}')
    print(f'lol: {lol}')
    print(f'pfs: {pf}')
    print(f'heights: {heights}')
    pf_order = 1
    ord_sum = len(hashkey)

    if lol:
        print("\nRemoving LOL orders . . . ", end = "")
        
        by_date = hashkey.groupby('date')['hashkey'].value_counts().to_frame()
    
        lol_items = []

        for ind, row in by_date.iterrows():
            if row['hashkey'] >= MIN_LOL_ORDERS:
                hash = ind[1]
                sum = 0
                tmp_items = []
                s_hash = hash.split(';')
                if len(s_hash) <= MAX_LOL_LINES:
                    for h in s_hash:
                        item, quant = h.split('*')
                        sum += int(quant)
                        tmp_items.append(item)

                    if sum <= MAX_LOL_ITEMS:
                        hashkey = hashkey.drop(hashkey[(hashkey['date'] == ind[0]) & (hashkey.hashkey == ind[1])].index)
                        for i in tmp_items:
                            if i not in lol_items:
                                lol_items.append(i) 
                        

        #print(hashkey.groupby('date')['hashkey'].value_counts().to_frame())
        print('Done')
        print('\nTotal Orders: {0:,}'.format(ord_sum))
        print('Orders Removed: {0:,} ({1:.2%})'.format(ord_sum - len(hashkey), (ord_sum - len(hashkey)) / ord_sum))
        print('Remaining: {0:,} ({1:.2%})'.format(len(hashkey), len(hashkey) / ord_sum))
        lol_items.sort()
        pf_info.append([cust, 'LOL', pf_order, (ord_sum - len(hashkey)) / ord_sum, lol_items])
        pf_order += 1

    #print(pf_info)
    print('\nBuilding Pickfaces . . . ', end = '')

    # Connect to DB
    cnxn = connect_db()

    #print(hashkey)
    order_count = hashkey.order_config.value_counts().to_frame()\
        .rename(columns={'order_config': 'order_count'})

    order_count['visited'] = False
    #print(order_count)
    #print(len(order_count[order_count.order_count == 1]))

    # Connect to DB to get item info
    item_sql = '''SELECT i.ASC_id AS item_id, i.description, i.case_qty, i.width, i.length, i.height
                    FROM Item AS i
                    WHERE i.customer = ucase('{0:s}');'''.format(cust)

    item_info = pd.read_sql(item_sql, cnxn).set_index('item_id')

    #print(item_info)

    cnxn.close()

    top = []        # Store top X items for each pickface
    pickfaces = []  # Store each pickface

    backup = []     # Incomplete order configurations to use for later

    ignore_num = 0

    if ignored:
        print('Removing orders with ignored items . . . ', end = '')
        for ind, row in order_count.iterrows():
            # If any items in the order_count index match, delete the configuration
            if any(x in ignored for x in ind.split(';')):
                ignore_num += row['order_count']
                order_count = order_count.drop(ind)
                
        print('Done')
        print('Orders Removed: {ignore_num:,} ({ignore_num/ord_sum:.2%})')

    # loop through each pickface number and get those top X items
    for p in range(len(pf)):
        slots = pf[p][2] * pf[p][1] * pf[p][0]
        configs = 0 # Number for tracking order configurations
        # each pickface is stored in its own dataframe
        top.append(pd.DataFrame(columns = ['item_id', 'orders', 'order_configs']))
        backup = []

        if required:
            for req in required:
                tmp = pd.DataFrame([[req, 0, []]], 
                                    columns = ['item_id', 'orders', 'order_configs'])
                top[p] = top[p].append(tmp, ignore_index = True)

        for ind, row in order_count.iterrows():
            items = ind.split(';')

            
            # if there aren't enough spaces to hold the next order configuration...
            # count the number of items missing and if they can fit in, add the config
            # and items
            if len(items) > (slots - len(top[p])):
                not_in = 0
                for i in items:
                    if i not in list(top[p]['item_id']):
                        not_in += 1

                if not_in > (slots - len(top[p])):
                    for i in items:
                        backup.append([i, row['order_count']])

                    continue

            # track if order configuration has been accounted for
            order_count.at[ind, 'visited'] = True
            h2 = []

            for i in items:

                # add the count if the item is already in the pickface
                if i in list(top[p]['item_id']):
                    #print(top[p])
                    ind_i = top[p][top[p]['item_id'] == i].index.values[0]
                    tmp = top[p].at[ind_i, 'orders']

                    top[p].at[ind_i, 'orders'] = tmp + row['order_count']
                    top[p].at[ind_i, 'order_configs'].append((str(configs), row['order_count']))

                else:
                    tmp = pd.DataFrame([[i, row['order_count'], [(str(configs), row['order_count'])]]], 
                                        columns = ['item_id', 'orders', 'order_configs'])
                    top[p] = top[p].append(tmp, ignore_index = True)
                    
            configs += 1

        print('')
        while len(top[p].index) < slots:
            
            if (backup[0][0] not in list(top[p].item_id)):
                print(f'incomplete order configuration: {backup[0][0]}')
                tmp = pd.DataFrame([backup[0].append((str(configs), 0))], columns = ['item_id', 'orders', 'order_configs'])
                top[p] = top[p].append(tmp, ignore_index = True)
            backup.pop(0)

            if len(backup) == 0:
                break


        # Calculate the percent of total orders served from the pickface
        sum = top[p].orders.sum()
        top[p]['percent'] = top[p].apply(lambda row: row.orders / sum, axis = 1)

        top[p] = top[p].sort_values('orders', ascending = False).set_index('item_id')
        print(f'\nTop {len(top[p])} Items:\n{top[p]}')
        top[p] = top[p].join(item_info, how = "left")
        #print(top[p])
        visited = list(order_count[order_count.visited == True].index)
        #print(visited)
        
        ord_serv = order_count[order_count.visited == True].order_count.sum()
        print('\nTotal Orders: {0:,}'.format(ord_sum))
        print('Ideal Conditions:')
        ord_per = ord_serv / ord_sum
        print('\t% Orders Served: {0:.2%}'.format(ord_per))

        sub_hashkey = hashkey[hashkey.order_config.isin(visited)]
        sub_val_count = sub_hashkey['date'].value_counts()

        d_min = int(round(sub_val_count.min()))
        q1 = int(round(np.nanpercentile(sub_val_count, 25)))
        d_med = int(round(sub_val_count.median()))
        q3 = int(round(np.nanpercentile(sub_val_count, 75)))
        d_max = int(round(sub_val_count.max()))

        print('\tOrders/Day:')
        print(f'\tMin = {d_min:,}')
        print(f'\t1Qt = {q1:,}')
        print(f'\tMed = {d_med:,}')
        print(f'\t3Qt = {q3:,}')
        print(f'\tMax = {d_max:,}')
        tmptmp = list(top[p].index)
        tmptmp.sort()
        #print(top[p])
        pf_info.append([cust, pf_switch(top[p]), pf_order, ord_per, tmptmp])
        pf_order += 1

        #print(pf_info)
        min_max = min_max_from_hashkey(sub_hashkey, item_info)
        #print(min_max)

        top[p] = top[p].join(min_max, how = "left")

        # Remove all the used order configurations
        order_count = order_count[order_count.visited != True]

        pickf = Pickface(pf[p][0], pf[p][1], pf[p][2], heights[p], 1, cust = cust)
        #pickfaces.append(Pickface(pf[p][0], pf[p][1], pf[p][2], heights[p], 1, cust = cust, row_priority = prior[p]))
        #pickf = switch_pf(pf[p], cust, 1, row_height)
        #print(top[p])
        pickf.populate(top[p])
        #pickf.display()
        #pickf.evaluate(hashkey)
        pickf.to_csv()
        pickfaces.append(copy.deepcopy(pickf))
        
        for b in range(pickf.bays):
            for r in range(pickf.bay_rows):
                for c in range(pickf.bay_cols):
                    item = pickf.slots[b][r][c]
                    #print(item)
                    if item is not None:
                        frame.append([cust,
                                      pf_switch(top[p]),
                                      f'{str(b+1).zfill(2)}.{ROWS[r]}{str(c+1).zfill(2)}',
                                      item.id, 
                                      item.desc,
                                      b + 1, 
                                      ROWS[r], 
                                      c + 1, 
                                      item.min, 
                                      item.max])

    

    remaining = set()
    ord_per = (order_count.order_count.sum() + ignore_num) / ord_sum

    if ignored:
        for i in ignored:
            remaining.add(i)

    #print(order_count)
    for ind, row in order_count.iterrows():
        for i in ind.split(';'):
            remaining.add(i)

    remain_list = list(remaining)
    remain_list.sort()
    pf_info.append([cust, 'Remaining', pf_order, ord_per, remain_list])

def evaluate(pfs, hashkey):
    '''Evaluate a pickface using a hashkey of orders.

    '''
    print('\nEvaluating pickfaces')
    ord_sum = len(hashkey)

    if type(pfs) is not list:
        pfs = [pfs]
    #hashkey = remove_lol(hashkey)
    order_count = hashkey.order_config.value_counts().to_frame()\
        .rename(columns={'order_config': 'order_count'})
    order_count['visited'] = False

    
    #by_date = hashkey['date'].value_counts().reset_index().rename(columns = {'date': 'count', 'index': 'date'}).sort_values('date').set_index('date')
    #by_date.plot(kind = 'bar', legend = None)
    #plt.show()

    for pf in pfs:
        items = pf.list_items()

        ord_serv = 0
        
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
        sub_hashkey = hashkey[hashkey.order_config.isin(visited)]
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

        #by_date = sub_val_count.to_frame().reset_index().rename(columns = {'date': 'count', 'index': 'date'}).sort_values('date').set_index('date')
        #by_date.plot(kind = 'bar', legend = None)
        #plt.show()

        order_count = order_count[~order_count.visited]

def single_single_analyze(hashkey_path, **kwargs):
    hashkey = load_hashkey(hashkey_path)
    hashkey['type'] = ''
    print(hashkey)

    single_items = set()
    single_single_items = []

    ignored = kwargs.get('ignore')

    ord_sum = len(hashkey)
    
    configs = hashkey['order_config'].value_counts().to_frame()

    if ignored:
        print('Removing orders with ignored items . . . ', end = '')
        for ind, row in configs.iterrows():
            # If any items in the configs index match, delete the configuration
            if any(x in ignored for x in ind.split(';')):
                hashkey = hashkey[hashkey['order_config'] != ind]

        print('Done')
        print(f'Orders Removed: {ord_sum - len(hashkey):,} ({(ord_sum - len(hashkey))/ord_sum:.2}')

    for ind, row in hashkey.iterrows():
        hash = row['hashkey'].split(';')

        if len(hash) == 1:
            item, qty = hash[0].split('*')
            if qty == '1':
                hashkey.at[ind, 'type'] = 'single'
                single_items.add(item)

        elif len(hash) == 2:
            sing_sing = False
            for h in hash:
                item, qty = h.split('*')
                if qty == '1':
                    sing_sing = True
                else:
                    sing_sing = False

                if not sing_sing:
                    break

            if sing_sing:
                hashkey.at[ind, 'type'] = 'single-single'
                for h in hash:
                    single_single_items.append(h.split('*')[0])

    singles = copy.deepcopy(hashkey[hashkey['type'] == 'single'])
    sing_sing = copy.deepcopy(hashkey[hashkey['type'] == 'single-single'])

    single_single_items = {i:single_single_items.count(i) for i in single_single_items}

    single_plus = max(single_single_items, key=single_single_items.get)
    print(f'Single-Single Item: {single_plus}')
    
    sing_sing['is_sing_plus'] = False

    for ind, row in sing_sing.iterrows():
        has_single_plus = False
        for hash in row['hashkey'].split(';'):
            if hash.split('*')[0] == single_plus:
                has_single_plus = True

        if has_single_plus:
            sing_sing.at[ind, 'is_sing_plus'] = True

    sing_sing = sing_sing[sing_sing['is_sing_plus']]


    print('No LOL removed')
    print(f'Total Orders: {ord_sum}')
    print('Single stats:')
    print(f'Single items: {single_items}')
    by_date = singles['date'].value_counts().to_frame().reset_index().rename(columns = {'date': 'count', 'index': 'date'}).sort_values('date').set_index('date')
    by_date.plot(kind = 'hist', legend = True, grid = True)
    plt.axvline(by_date['count'].median(), color='k', linestyle='dashed')
    print(f'Orders: {len(singles):,} ({len(singles)/ord_sum:.2%})')
    plt.show()

    print('\nSingle-Single stats:')
    print(f'Single +1 item: {single_plus}')
    by_date = sing_sing['date'].value_counts().to_frame().reset_index().rename(columns = {'date': 'count', 'index': 'date'}).sort_values('date').set_index('date')
    by_date.plot(kind = 'hist', legend = True, grid = True)
    plt.axvline(by_date['count'].median(), color='k', linestyle='dashed')
    print(f'Orders: {len(sing_sing):,} ({len(sing_sing)/ord_sum:.2%})')
    plt.show()

    hashkey = remove_lol(hashkey)

    hashkey['type'] = ''
    #print(hashkey)

    single_items = set()
    single_single_items = []

    configs = hashkey['hashkey'].value_counts()

    for ind, row in hashkey.iterrows():
        hash = row['hashkey'].split(';')

        if len(hash) == 1:
            item, qty = hash[0].split('*')
            if qty == '1':
                hashkey.at[ind, 'type'] = 'single'
                single_items.add(item)

        elif len(hash) == 2:
            sing_sing = False
            for h in hash:
                item, qty = h.split('*')
                if qty == '1':
                    sing_sing = True
                else:
                    sing_sing = False

                if not sing_sing:
                    break

            if sing_sing:
                hashkey.at[ind, 'type'] = 'single-single'
                for h in hash:
                    single_single_items.append(h.split('*')[0])

    singles = copy.deepcopy(hashkey[hashkey['type'] == 'single'])
    sing_sing = copy.deepcopy(hashkey[hashkey['type'] == 'single-single'])

    single_single_items = {i:single_single_items.count(i) for i in single_single_items}

    single_plus = max(single_single_items, key=single_single_items.get)
    print(f'Single-Single Item: {single_plus}')
    
    sing_sing['is_sing_plus'] = False

    for ind, row in sing_sing.iterrows():
        has_single_plus = False
        for hash in row['hashkey'].split(';'):
            if hash.split('*')[0] == single_plus:
                has_single_plus = True

        if has_single_plus:
            sing_sing.at[ind, 'is_sing_plus'] = True

    sing_sing = sing_sing[sing_sing['is_sing_plus']]


    print('LOL removed')
    print(f'Total Orders: {ord_sum}')
    print('Single stats:')
    print(f'Single items: {single_items}')
    by_date = singles['date'].value_counts().to_frame().reset_index().rename(columns = {'date': 'count', 'index': 'date'}).sort_values('date').set_index('date')
    by_date.plot(kind = 'hist', legend = True, grid = True)
    plt.axvline(by_date['count'].median(), color='k', linestyle='dashed')
    print(f'Orders: {len(singles):,} ({len(singles)/ord_sum:.2%})')
    plt.show()

    print('\nSingle-Single stats:')
    print(f'Single +1 item: {single_plus}')
    by_date = sing_sing['date'].value_counts().to_frame().reset_index().rename(columns = {'date': 'count', 'index': 'date'}).sort_values('date').set_index('date')
    by_date.plot(kind = 'hist', legend = True, grid = True)
    plt.axvline(by_date['count'].median(), color='k', linestyle='dashed')
    print(f'Orders: {len(sing_sing):,} ({len(sing_sing)/ord_sum:.2%})')
    plt.show()



#single_single(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Desktop\batch_1842724.csv")


#single_single_analyze(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\Slotting\Slotting\data\PURA_ATL_hashkey.csv",
#                      ignore = ['1DB'])
#single_single_analyze(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\Slotting\Slotting\data\pura_wj_hashkey.csv",
#                      ignore = ['1DB'])

print('Loading all client PF info . . . ', end = '')

custs_info = pd.read_excel(r"C:\Users\David.Moreno\OneDrive - Visible SCM\Coding\Slotting\Slotting\data\custs_info.xlsx",
                           dtype = 'string')
print('Done')

df_pf_info = [] #pd.DataFrame(columns=['client','type','order','percent','items'])
df_frame = []


print(custs_info)
threads = []

for ind, row in custs_info.iterrows():
    slotting(row, df_frame, df_pf_info)
    #t = threading.Thread(target = slotting, args = (row, df_frame, df_pf_info))
    #t.start()
    #threads.append(t)

#for t in threads:
    #t.join()


pfs_info = pd.DataFrame(df_pf_info, columns = ['client','type','order','percent','items'])
print(pfs_info)

columns = ['client', 'pickface', 'location', 'item_id', 'desc', 'bay', 'row', 
           'col', 'min', 'max']
pfs = pd.DataFrame(df_frame, columns = columns)

print(pfs)

