import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_row', None)
import numpy as np
import datetime
from Pickface import *
from DB import connect_db
from Hashkey import *



def switch_pf(i, cust, depth, height):
    '''A Function meant to server like a switch statement, creates the correct pickface based on input value

    '''
    if i == 9:
        return PF_9(cust, depth, height)

    elif i == 27:
        return PF_27(cust, depth, height)

    elif i == 32:
        return PF_32(cust, depth, height)

    elif i == 48:
        return PF_48(cust, depth, height)

    else:
        return Omni(cust, depth, height)


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

        pickf = switch_pf(pf[p], cust, 1, row_height)
        print(top[p])
        pickf.populate(top[p])
        pickf.display()
        pickf.evaluate(hashkey)
        pickf.to_csv()
        pickfaces.append(pickf)

    return pickfaces



def slotting(hashkey, pf, cust, row_height, **kwargs):
    '''Builds the desired pickfaces based on the most popular order configurations.

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
    #print(order_count)

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

        pickf = switch_pf(pf[p], cust, 1, row_height)
        print(top[p])
        pickf.populate(top[p])
        pickf.display()
        pickf.evaluate(hashkey)
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



def single_single(hashkey, **kwargs):
    '''Find statistics on the single-single orders in a hashkey.

    '''
    # Get ignored or required items
    ignored = kwargs.get('ignore')

    #print(hashkey)
    hashkey_count = hashkey.hashkey.value_counts().to_frame()
    hashkey_count['visited'] = False
    #print(hashkey_count)

    ord_sum = hashkey_count.hashkey.sum()
    sing_sum = 0

    if ignored:
        hashkey_count['single'] = False

        for index, row in hashkey_count.iterrows():
            hash = index.split(';')
            length = len(hash)
            single = True

            for h in hash:
                i = h.split('*')

                if i[0] in ignored:
                    length -= 1
                    single = False

                elif i[-1] != '1':
                    length = -1
                    break

            if length == 1:
                hashkey_count.at[index, 'visited'] = True
                hashkey_count.at[index, 'single'] = single

    else:
        for index, row in hashkey_count.iterrows():
            if len(index.split(';')) == 1:
                if index.split('*')[-1] == '1':
                    hashkey_count.at[index, 'visited'] = True
    
    singles = hashkey_count[hashkey_count.visited]
    print(singles)
    print('\nTotal Singles: {0:,} ({1:.2%})'.format(singles.hashkey.sum(), 
                                                  singles.hashkey.sum()/hashkey_count.hashkey.sum()))
    sing_pls = singles[~singles.single].hashkey.sum()
    print('\tSingle + 1: {0:,} ({1:.2%})'.format(sing_pls, sing_pls/singles.hashkey.sum()))
    sing_sing = singles[singles.single].hashkey.sum()
    print('\tSingle-Single: {0:,} ({1:.2%})'.format(sing_sing, sing_sing/singles.hashkey.sum()))
    visited = list(singles.index)

    sub_hashkey = hashkey[hashkey.hashkey.isin(visited)]
    sub_val_count = sub_hashkey['date'].value_counts()

    sing_sum = len(sub_hashkey)
    sing_per = sing_sum / ord_sum
    print('\nTotal Orders: {0:,}'.format(ord_sum))
    print('Single-Single Orders: {0:,} ({1:.2%})'.format(sing_sum, sing_per))

    min = int(round(sub_val_count.min()))
    q1 = int(round(np.nanpercentile(sub_val_count, 25)))
    med = int(round(sub_val_count.median()))
    q3 = int(round(np.nanpercentile(sub_val_count, 75)))
    max = int(round(sub_val_count.max()))

    print('Orders/Day:')
    print(f'\tMin = {min:,}')
    print(f'\t1Qt = {q1:,}')
    print(f'\tMed = {med:,}')
    print(f'\t3Qt = {q3:,}')
    print(f'\tMax = {max:,}')

    return