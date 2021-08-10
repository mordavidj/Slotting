import pandas as pd
pd.set_option('max_columns', None)
#pd.set_option('max_row', None)
import numpy as np
import datetime as dt
from Pickface import *
from DB import connect_db
from Hashkey import *
import pyodbc
import os
#import copy

POD = [1, 3, 3]
HVPNP = [3, 3, 3]
PNP = [4, 3, 4]

MIN_LOL_ORDERS = 50
MAX_LOL_LINES = 3
MAX_LOL_ITEMS = 5


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



def pf_switch(items):
    if len(items) <= 9:
        return 'POD'

    elif len(items) <= 27:
        return 'HVPnP'

    elif len(items) <= 48:
        return 'PnP'

    else:
        return 'Omni'



def pf_dim_switch(pf: str):
    if pf == 'POD':
        return POD
    elif pf == 'HVPnP':
        return HVPNP
    elif pf == 'PnP':
        return PNP
    else:
        return [int(x) for x in pf.split(',')]



def slotting(info: pd.Series, frame: list, pf_info: list, pf_items: list, 
             old_pfs_info: pd.DataFrame = None, old_pfs: pd.DataFrame = None):
    '''Take customer info and Pickface dimensions and find an optimized collection to handle the most number of orders per pickface

    '''
    cust = info['client']

    warehouse = info['warehouse']
    # A hashkey that contains all the order data
    hashkey = None
    if type(info['hashkey']) is pd.DataFrame:
        hashkey = info['hashkey']
    elif type(info['hashkey']) is str:
        hashkey = pd.read_excel(info['hashkey'])

    else:
        hashkey = get_hashkey(client = cust, warehouse = warehouse)

    lol = True if info['lol'].upper() == 'TRUE' else False

    lol_ignore = str(info['lol_ignore']).split(';')\
        if not pd.isna(info['lol_ignore']) else None

    single = True if info['single'].upper() == 'TRUE' else False

    single_ignore = str(info['single_ignored']).split(';')\
        if not pd.isna(info['single_ignore']) else None

    pf = [pf_dim_switch(p) for p in info['pfs'].split(';')]

    h1 = info['pfs_heights'].split(';')
    heights = []
    for h2 in h1:
        heights.append([int(h) for h in h2.split(',')])

    ignored = str(info['pfs_ignore']).split(';')\
        if not pd.isna(info['pfs_ignore']) else None
    required = str(info['pfs_require']).split(';')\
        if not pd.isna(info['pfs_require']) else None
    
    print(f'client: {cust}')
    print(f'lol: {lol}')
    if lol and lol_ignore:
        print(f'lol_ignore: {lol_ignore}') 
    print(f'single: {single}')
    if single and single_ignore:
        print(f'single_ignore: {single_ignore}') 
    if ignored:
        print(f'pfs_ignore: {ignored}')
    if required:
        print(f'pfs_require: {required}')
    print(f'pfs: {pf}')
    print(f'heights: {heights}')
    pf_order = 1
    ord_sum = len(hashkey)

    # Remove any orders that qualify under the conditions specified by the schedulers
    # Usually: 50+ orders with less than 3 different items summing less than 5 total
    if lol:
        
        print("\nRemoving LOL orders . . . ", end = "")
        
        hashkey['d'] = pd.to_datetime(
            hashkey.apply(lambda row: pd.to_datetime(row['date']) + dt.timedelta(days=1) if row['date'].hour >= 12 else row['date'], 
                              axis = 1))\
                                  .dt.date
    
        hashkey['s'] = hashkey.apply(lambda row: 1 if row['date'].hour >= 12 or row['date'].hour < 5 else 2, axis = 1)
        lol_hashkey = pd.DataFrame().reindex_like(hashkey)

        by_schedule = hashkey.groupby(['d', 's'])['hashkey'].value_counts().to_frame()
    
        lol_items = set()

        for ind, row in by_schedule.iterrows():
            if row['hashkey'] >= MIN_LOL_ORDERS:
                hash = ind[2]
                sum = 0
                tmp_items = []
                s_hash = hash.split(';')

                if len(s_hash) <= MAX_LOL_LINES:
                    for h in s_hash:
                        item, quant = h.split('*')
                        if lol_ignore and item in lol_ignore:
                            sum = MAX_LOL_ITEMS + 1
                            break
                        sum += int(quant)
                        tmp_items.append(item)

                    if sum <= MAX_LOL_ITEMS:
                        lol_hashkey = lol_hashkey.append(
                            hashkey[(hashkey['d'] == ind[0])\
                                & (hashkey.s == ind[1])\
                                & (hashkey.hashkey == hash)])

                        hashkey = hashkey.drop(
                            hashkey[(hashkey['d'] == ind[0])\
                                & (hashkey.s == ind[1])\
                                & (hashkey.hashkey == hash)].index)

                        for i in tmp_items:
                            lol_items.add(i) 
                        

        #print(hashkey.groupby('date')['hashkey'].value_counts().to_frame())
        print('Done')
        print('\nTotal Orders: {0:,}'.format(ord_sum))
        print('Orders Removed: {0:,} ({1:.2%})'.format(
            ord_sum - len(hashkey), (ord_sum - len(hashkey)) / ord_sum))

        lol_items = list(lol_items)
        lol_df = lol_hashkey['hashkey'].value_counts().to_frame()
        #print(lol_df)
        lol_df['percent'] = lol_df.apply(lambda row:\
            (row['hashkey'] / lol_df['hashkey'].sum()) * 100, axis = 1)
        for ind, row in lol_df.iterrows():
            pf_items.append([cust, warehouse, 'LOL', ind, row['hashkey'], 0])
        #print(pf_items)
        ord_per = (ord_sum - len(hashkey)) / ord_sum
        pf_info.append([cust, warehouse, 'LOL', pf_order, ord_per, f'{ord_per:.2%}', 
                        lol_items])
        pf_order += 1
        hashkey = hashkey.drop(columns=['d','s'])

    hashkey['date'] = pd.to_datetime(hashkey['date']).dt.date

    # Orders that are a single item.
    if single:
        print('\nRemoving Single-Single orders . . . ', end = '')
        hashkey['is_single'] = False

        for ind, row in hashkey.iterrows():
            hash = row['hashkey'].split(';')

            if len(hash) == 1:
                item, qty = hash[0].split('*')
                if qty == '1':
                    hashkey.at[ind, 'is_single'] = True

        singles = copy.deepcopy(hashkey[hashkey['is_single']])
        hashkey = hashkey[~hashkey['is_single']]
        hashkey = hashkey.drop(columns = ['is_single'])
        print('Done')

        sing_df = singles['order_config'].value_counts().to_frame()
        single_items = sing_df.index.tolist()
        sing_df['percent'] = sing_df.apply(
            lambda row: (row['order_config'] / sing_df['order_config'].sum()) * 100,
           axis = 1)
        for ind, row in sing_df.iterrows():
            pf_items.append([cust, warehouse, 'Single', ind, row['percent']])

        ord_per = len(singles) / ord_sum
        print(f'\nSingle-Single Orders: {len(singles):,} ({ord_per:.2%})')
        pf_info.append([cust, warehouse, 'Single', pf_order, ord_per, f'{ord_per:.2%}', 
                        single_items])
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
    item_sql = '''SELECT i.ASC_id AS item_id, i.description, i.case_qty, 
                    i.width, i.length, i.height
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
        print(f'Orders Removed: {ignore_num:,} ({ignore_num/ord_sum:.2%})')

    # loop through each pickface number and get those top X items
    for p in range(len(pf)):
        slots = pf[p][2] * pf[p][1] * pf[p][0]
        configs = 0 # Number for tracking order configurations
        # each pickface is stored in its own dataframe
        top.append(pd.DataFrame(columns = ['item_id', 'orders', 
                                           'order_configs']))
        backup = []

        if required:
            for req in required:
                tmp = pd.DataFrame([[req, 0, []]], 
                                    columns = ['item_id', 'orders', 
                                               'order_configs'])
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
                    top[p].at[ind_i, 'order_configs']\
                        .append((str(configs), row['order_count']))

                else:
                    tmp = pd.DataFrame([[i, row['order_count'], 
                                         [(str(configs), row['order_count'])]]], 
                                        columns = ['item_id', 'orders', 
                                                   'order_configs'])
                    top[p] = top[p].append(tmp, ignore_index = True)
                    
            configs += 1

        print('')
        while len(top[p].index) < slots:
            
            if len(backup) == 0:
                break
            
            if (backup[0][0] not in list(top[p].item_id)):
                print(f'incomplete order configuration: {backup[0][0]}')
                tmp = pd.DataFrame([backup[0].append((str(configs), 0))], 
                                   columns = ['item_id', 'orders', 
                                              'order_configs'])
                top[p] = top[p].append(tmp, ignore_index = True)
            backup.pop(0)


        # Calculate the percent of total orders served from the pickface
        sum = top[p].orders.sum()
        top[p]['percent'] = top[p].apply(lambda row: row.orders / sum, 
                                         axis = 1)
        

        top[p] = top[p].sort_values('orders', ascending = False)\
            .set_index('item_id')
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
        pf_info.append([cust, warehouse, pf_switch(top[p]), pf_order, 
                        ord_per, f'{ord_per:.2%}', tmptmp])
        pf_order += 1

        if(old_pfs is not None):
            print(old_pfs)
            old_pfs_2 = old_pfs.loc[(cust, warehouse, pf_switch(top[p]))]
            old_top = old_pfs_2.item_id.tolist()
            print(old_top)
            print(tmptmp)
            new = [i for i in tmptmp if i not in old_top]
            old = [i for i in old_top if i not in tmptmp]

            if len(new) is not len(old):
                d = 1
                pass


            print(f'new: {new}')
            print(f'old: {old}')

            # If there are no differences, leave the arrangement alone
            if len(new) == 0:
                pf_type = pf_switch(top[p])
                for ind, row in old_pfs_2.iterrows():
                    frame.append([cust,
                                  warehouse,
                                    pf_type,
                                    ind,
                                    row['item_id'],
                                    row['desc'],
                                    row['bay'],
                                    row['row'],
                                    row['col'],
                                    row['min'],
                                    row['max'],
                                    0])

            

            else:
                # If there are too many differences, just redo the whole thing.
                if len(new) >= (len(top[p]) * (2 / 3)):
                    #print(pf_info)
                    min_max = min_max_from_hashkey(sub_hashkey, item_info)
                    #print(min_max)

                    top[p] = top[p].join(min_max, how = "left")

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
                                    frame.append([cust,
                                                  warehouse,
                                                  pf_switch(top[p]),
                                                  f'{str(b+1).zfill(2)}.{ROWS[r]}{str(c+1).zfill(2)}',
                                                  item.id, 
                                                  item.desc,
                                                  b + 1, 
                                                  ROWS[r], 
                                                  c + 1, 
                                                  item.min, 
                                                  item.max,
                                                  2])
                            
                else:
                    little_pf = old_pfs.loc[(cust, warehouse, pf_switch(top[p]))]
                    #print(little_pf)
                    open_locs = [i for i in little_pf[little_pf.item_id.isin(old)].index]
                    #print(open_locs)
                    tmp_pf = copy.deepcopy(little_pf)

                    for i in range(len(new)):
                        tmp_pf.at[open_locs[i], 'item_id'] = new[i]
                        #tmp_pf.at[open_locs[i], 'desc'] = item_info[item_info.]

                    #print(tmp_pf)
                    #print(top[p])
                    tmp_top = top[p].reset_index()[['item_id', 'percent']]
                    #print(tmp_top)
                    tmp_pf = tmp_pf.reset_index().merge(tmp_top, on = 'item_id').set_index('location')
                    #print(tmp_pf)
                    tmp_sum = tmp_pf.groupby('bay').agg({'percent':'sum'})
                    
                    if (tmp_sum.percent.max() - tmp_sum.percent.min()) > 0.05:
                        print(tmp_pf)
                        print(tmp_sum)
                        pass

                    min_max = min_max_from_hashkey(sub_hashkey, item_info)

                    #print(min_max)
                    tmp_pf['change'] = 0

                    for ind, row in tmp_pf.iterrows():
                        tmp_pf.at[ind, 'min'] = min_max.at[row['item_id'], 'min']
                        tmp_pf.at[ind, 'max'] = min_max.at[row['item_id'], 'max']
                        tmp_pf.at[ind, 'desc'] = item_info.at[row['item_id'], 'description']
                        if row['item_id'] in new:
                            tmp_pf.at[ind, 'change'] = 2

                        elif little_pf.at[ind, 'item_id'] is not tmp_pf.at[ind, 'item_id']:
                            tmp_pf.at[ind, 'change'] = 1

                    #print(tmp_pf)

                    for ind,row in tmp_pf.iterrows():
                        frame.append([cust,
                                      warehouse,
                                      pf_switch(top[p]),
                                      ind,
                                      row['item_id'],
                                      row['desc'],
                                      row['bay'],
                                      row['row'],
                                      row['col'],
                                      row['min'],
                                      row['max'],
                                      row['change']])

                    #print(frame)


                
        else:
            #print(pf_info)
            min_max = min_max_from_hashkey(sub_hashkey, item_info)
            #print(min_max)

            top[p] = top[p].join(min_max, how = "left")

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
                            frame.append([cust,
                                          warehouse, 
                                            pf_switch(top[p]),
                                            f'{str(b+1).zfill(2)}.{ROWS[r]}{str(c+1).zfill(2)}',
                                            item.id, 
                                            item.desc,
                                            b + 1, 
                                            ROWS[r], 
                                            c + 1, 
                                            item.min, 
                                            item.max,
                                            2])

        order_count = order_count[order_count.visited != True]

    remaining = set()
    ord_per = (order_count.order_count.sum() + ignore_num) / ord_sum

    if ignored:
        for i in ignored:
            remaining.add(i)

    print(order_count)
    remain_hashkey = hashkey[hashkey['order_config'].isin(order_count.index)]
    print(remain_hashkey)
    df = pd.concat([pd.Series(row['date'], row['hashkey'].split(';'))\
        for ind, row in remain_hashkey.iterrows()]).reset_index()
    print(df)
    df = df.rename(columns = {'index': 'hashkey', 0: 'date'})
    df = df[df['hashkey'] != '']
    print(df)
    # Split the hashkey into items and quantities
    df[['item', 'qty']] = df['hashkey'].str.split('*', expand=True)
    print(df)
    try:
        df['qty'] = df['qty'].astype('float')
    except:
        for ind, row in df.iterrows():
            try:
                float(row['qty'])
            except Exception as err:
                print(ind)
                print(row)
                print('')

    # Get the total shipped
    df = df.groupby('item').agg({'qty': 'sum'})
    #print(df)

    for ind, row in order_count.iterrows():
        for i in ind.split(';'):
            remaining.add(i)

    remain_list = list(remaining)
    remain_df = pd.DataFrame({'item':remain_list}).set_index('item')
    remain_df['order_count'] = 0
    for ind, row in order_count.iterrows():
        for i in ind.split(';'):
            remain_df.at[i, 'order_count'] += row['order_count']

    tmp_ord_sum = remain_df['order_count'].sum()
    remain_df['percent'] = remain_df\
        .apply(lambda row: (row['order_count'] / tmp_ord_sum) * 100, axis = 1)
    remain_df = remain_df.join(df)
    print(remain_df)
    for ind, row in remain_df.iterrows():
        pf_items.append([cust, warehouse, 'Omni', ind, row['order_count'], row['qty']])
    #print(df_items)
    pf_info.append([cust, warehouse, 'Omni', pf_order, ord_per, f'{ord_per:.2%}', 
                    remain_list])