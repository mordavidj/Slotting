import pandas as pd
import numpy as np
import datetime
from Pickface import *

#################################################
# Load a Power BI Hashkey report from a file and generate order configurations
#################################################
def load_powerBI_hashkey(filepath):
    print(f'\nLoading Power BI Hashkey: {filepath}')

    df = pd.read_csv(filepath,
                     dtype = 'string')\
                         .rename(columns = {'Created Date': 'date',
                                            'Optimization Hash Key': 'hashkey'})
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['order_config'] = ''

    # Order configurations are hashkeys without quantities
    print('Generating Order Configurations... ', end = '')
    for ind, row in df.iterrows():
        hashes = row['hashkey'].split(';')
        config = ''
        prefix = ''
        for hash in hashes:
            if hash != '':
                config += str(prefix + hash.split('*')[0])
                prefix = ';'

        df.at[ind, 'order_config'] = config

    print('Done')
    #print(df.head(20))
    #print(df.dtypes)
    return df


#################################################
# Load a hashkey that has already been generated
#################################################
def load_hashkey(filepath):
    print(f'\nLoading Power BI Hashkey: {filepath}')

    df = pd.read_csv(filepath,
                     dtype = 'string')

    df['date'] = pd.to_datetime(df['date']).dt.date

    # Create order configuration if not already in dataframe
    if 'order_config' not in df.columns:
        df['order_config'] = ''
        print('Generating Order Configurations... ', end = '')

        for ind, row in df.iterrows():
            hashes = row['hashkey'].split(';')
            config = ''
            prefix = ''
            for hash in hashes:
                if hash != '':
                    config += str(prefix + hash.split('*')[0])
                    prefix = ';'

            df.at[ind, 'order_config'] = config

    print('Done')
    return df


#################################################
# Take Order numbers, items, and quantities to build a hash key
# Use this if each item from an order is on its own line
#################################################
def generate_hashkey(df):
    print('\nGenerating hashkey from dataframe...', end = '')
    # Join each item and its quantity into a string on the same row
    df['hashkey'] = df.apply(lambda row: row.item_id + '*' + str(row.quantity), axis = 1)
    
    # Join all rows of an order into a hashkey string
    df = df.groupby(['order_number'])['hashkey'].apply(lambda row: ';'.join(row)).reset_index()
    print('Done')
    return df


#################################################
# Take a file from ASC and generate hashkeys and order configs
#################################################
def generate_hashkey_ASC(filepath):

    print(f'\nGenerating Hashkey from ASC file: {filepath}')

    hash = ''
    config = ''
    prefix = ''     
    n_frame = []    # Stores all items in the desired format
    n_row = []      # Keeps each order together in one row

    for index, row in df.iterrows():
        
        # If the it item can be converted into a date, it's a new item
        try:
            date_time = datetime.datetime.strptime(row['SHIPDATE'], "%m/%d/%Y %I:%M:%S %p")
            n_row.append(hash)
            n_row.append(config)
            n_frame.append(n_row.copy())
            hash = ''
            config = ''
            prefix = ''
            
            #print(n_frame)
            n_row.clear()
            n_row.append(date_time)
            n_row.append(row['ORDERNUMBER'])

        except:
            hash += prefix + str(row["ORDERNUMBER"]) + '*' + str(row["SHIPDATE"])
            config += prefix + str(row["ORDERNUMBER"])
            prefix = ';'

    n_row.append(str)
    tmp = n_row
    #print(tmp)
    n_frame.append(n_row.copy())

    # first item is a blank line, so pop it
    n_frame.pop(0)
    hashkey = pd.DataFrame(n_frame, columns = ['date', 'order_number', 'hashkey', 'order_config'])

    print('Done')
    return hashkey


#################################################
# Take an exported file from the SQL server and generate hashkeys and order configs
# WARNING: SQL server puts 0's when an order shipped incompletely
#################################################
def generate_hashkey_SQL(filepath):
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


#################################################
# Calculate a min-max from a hashkey using 80th and 90th 
# percentiles of items shipped each day as min and max
# and using master case quantities.
#################################################
def min_max_from_hashkey(hashkey, case_info):
    print('Calculating Min-Max from hashkey...', end = '')

    # Split hashkey by each item while keeping the date
    df = pd.concat([pd.Series(row['date'], row['hashkey'].split(';')) for _, row in hashkey.iterrows()]).reset_index()
    df = df.rename(columns = {'index': 'hashkey', 0: 'date'})
    df = df[df['hashkey'] != '']
    #print(df.head())
    
    # Split the hashkey into items and quantities
    df[['item_id', 'qty']] = df['hashkey'].str.split('*', expand=True)
    df['qty'] = df['qty'].astype(int)
    #print(df.head())

    # Get the total shipped each day
    df = df.groupby(['date', 'item_id']).agg({'qty': 'sum'})
    df.reset_index(inplace=True)
    #print(df.head())

    # Pivot so that day sums of all item quantities are aligned on the same row with the date
    pivot_df = df.pivot(index = 'date', columns = 'item_id', values = 'qty')
    min_max = pd.DataFrame(columns = ['item_id', '80', '90'])

    for i in range(1, len(pivot_df.columns)):
        min_max = min_max.append({'item_id': pivot_df.columns[i],
                                  '80': np.nanpercentile(pivot_df[pivot_df.columns[i]], 80), 
                                  '90': np.nanpercentile(pivot_df[pivot_df.columns[i]], 90)}, 
                                 ignore_index = True)

    #print(min_max.head())

    min_max = min_max.set_index('item_id').join(case_info.set_index('item_id'), how='left')

    min_max['min'] = min_max.apply(lambda row: to_int(np.ceil(row['80'] / row['case_qty'])), axis = 1)
    min_max['max'] = min_max.apply(lambda row: to_int(np.ceil(row['90'] / row['case_qty'])), axis = 1)
    #print(min_max.head())
    return min_max


#################################################
# Format NaN's, None's, and others into integers
#################################################
def to_int(val):
    if math.isnan(val) or val is None:
        return ''
    else:
        return int(val)


#################################################
# A Function meant to server like a switch statement,
# creates the correct pickface based on input value
#################################################
def switch_pf(i, cust, depth, height):
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


#################################################
# Builds the desired pickfaces based on the most popular order configurations.
# It is continuous because no items will repeat across any of the pickfaces,
# incomplete orders get passed to the next pickface.
#################################################
def continuous_slotting(hashkey, pf, cust):
    # convert the type to an iterable list
    if type(pf) is not list:
        pf = [pf]

    # Convert each item in pickface list to integer
    for i in range(len(pf)):
        try:
            pf[i] = int(pf[i])
        except:
            raise TypeError(f'Pickface number {i + 1} was not valid.')

    if cust == '':
        raise Exception('"cust" is required for documentation purposes.')
    
    # Create a table of the most popular order configurations.
    order_count = hashkey.order_config.value_counts().to_frame()\
        .rename(columns={'order_config': 'order_count'})
    order_count['visited'] = False

    # Create a set of all the unique item id's for using across pickfaces
    item_ids = set()
    for index, row in order_count.iterrows():
        for i in index.split(';'):
            item_ids.add(i)
    items_df = pd.DataFrame(item_ids, columns = ['item_id'])\
        .sort_values('item_id')\
        .set_index('item_id')

    items_df['used'] = False
    print(items_df)

    print(order_count)


    # Load the master case quantities for min-max calculations
    #case_info = pd.read_excel('../../../Documents/Master Case info.xlsx',
    #                     sheet_name = 'Case')[['item_id', 'case_qty']]


    top = []        # Store top X items for each pickface
    pickfaces = []  # Store each pickface
    # Order sums for statistics reporting
    ord_sum = order_count.order_count.sum()
    backup = []     # Incomplete order configurations to use for later
    configs = 0     # Number for tracking order configurations

    # loop through each pickface number and get those top X items
    for p in range(len(pf)):
        
        # each pickface is stored in its own dataframe
        top.append(pd.DataFrame(columns = ['item_id', 'orders', 'order_configs']))
        if backup:
            # Add all the popular items from the last pass to this one
            pass

        for ind, row in order_count.iterrows():
            items = ind.split(';')

            # if there aren't enough spaces to hold the next order configuration, store it and skip for now
            if len(items) > (pf[p] - len(top[p])):
                for i in items:
                    backup.append([i, row['order_count'], (configs, row['order_count'])])
                
                configs += 1
                continue

            # track if order configuration has been accounted for
            order_count.at[ind, 'visited'] = True

            for i in items:
                # if there are still open slots...
                if len(top[p].index) < pf[p]:

                    # add the count if the item is already in the pickface
                    if i in list(top[p]['item_id']):
                        #print(top[p])
                        ind_i = top[p][top[p]['item_id'] == i].index.values[0]
                        tmp = top[p].at[ind_i, 'orders']

                        top[p].at[ind_i, 'orders'] = tmp + row['order_count']
                        top[p].at[ind_i, 'order_configs'].append((str(configs), row['order_count']))

                    else:
                        tmp = pd.DataFrame([[i, row['order_count'], [(str(configs), row['order_count'])]]], columns = ['item_id', 'orders', 'order_configs'])
                        top[p] = top[p].append(tmp, ignore_index = True)

                else:
                    break

            configs += 1

            # break the loop if there are no more slots
            if len(top[p].index) >= pf[p]:
                break

        print('')
        while len(top[p].index) < pf[p]:
            
            if (backup[0][0] not in list(top[p].item_id)):
                print(f'incomplete order configuration: {backup[0][0]}')
                tmp = pd.DataFrame([backup[0].append((str(configs), 0))], columns = ['item_id', 'orders', 'order_configs'])
                top[p] = top[p].append(tmp, ignore_index = True)
            backup.pop(0)

            if len(backup) == 0:
                break

        # Calculate the percent of orders per item within the pickface
        sum = top[p].orders.sum()
        top[p]['percent'] = top[p].apply(lambda row: row.orders / sum, axis = 1)

        top[p] = top[p].sort_values('orders', ascending = False).set_index('item_id')
        print(f'\nTop {len(top[p])} Items:\n{top[p]}')
        #print(top[p])
        #print(top[p]['percent'].sum())
        #print(order_count)

        visited = list(order_count[order_count.visited == True].index)
        #print(visited)
        
        ord_serv = order_count[order_count.visited == True].order_count.sum()
        print(f'\nTotal Orders: {ord_sum}')
        print(f'Orders Served by PF: {ord_serv}')
        ord_per = ord_serv / ord_sum
        print('% Orders Served: {:.2%}'.format(ord_per))

        sub_hashkey = hashkey[hashkey.order_config.isin(visited)]
        #print(sub_hashkey)
        #min_max = min_max_from_hashkey(sub_hashkey, case_info)
        #print(min_max)
        
        # Remove all the used order configurations
        order_count = order_count[order_count.visited != True]

        # Save the pickface info to a csv
        #top[p].to_csv(f'{cust}-{pf[p]}_slots.csv')

        pickface = switch_pf(pf[p], cust, 1, 15)
        #print(top[p])
        pickface.populate(top[p])
        pickface.display()
        pickface.to_csv()
        pickfaces.append(pickface)

    return pickfaces


#################################################
# Builds the desired pickfaces based on the most popular order configurations.
#################################################
def slotting(hashkey, pf, cust):

    # convert the type to an iterable list
    if type(pf) is not list:
        pf = [pf]

    # Convert each item in pickface list to integer
    for i in range(len(pf)):
        try:
            pf[i] = int(pf[i])
        except:
            raise TypeError(f'Pickface number {i + 1} was not valid.')

    if cust == '':
        raise Exception('"cust" is required for documentation purposes.')
    
    #print(hashkey)
    order_count = hashkey.order_config.value_counts().to_frame()\
        .rename(columns={'order_config': 'order_count'})
    order_count['visited'] = False

    hashkey_count = hashkey.hashkey.value_counts().to_frame()
    print(order_count)

    # Load the master case quantities for min-max calculations
    #case_info = pd.read_excel('../../../Documents/Master Case info.xlsx',
    #                     sheet_name = 'Case')[['item_id', 'case_qty']]


    top = []        # Store top X items for each pickface
    pickfaces = []  # Store each pickface
    # Order sums for statistics reporting
    ord_sum = order_count.order_count.sum()
    backup = []     # Incomplete order configurations to use for later

    
    # loop through each pickface number and get those top X items
    for p in range(len(pf)):
        configs = 0 # Number for tracking order configurations
        # each pickface is stored in its own dataframe
        top.append(pd.DataFrame(columns = ['item_id', 'orders', 'order_configs']))
        backup = []

        for ind, row in order_count.iterrows():
            items = ind.split(';')

            # if there aren't enough spaces to hold the next order configuration, store it and skip for now
            if len(items) > (pf[p] - len(top[p])):
                for i in items:
                    backup.append([i, row['order_count']])
                
                continue

            # track if order configuration has been accounted for
            order_count.at[ind, 'visited'] = True
            h2 = []

            for i in items:
                # if there are still open slots...
                if len(top[p].index) < pf[p]:

                    # add the count if the item is already in the pickface
                    if i in list(top[p]['item_id']):
                        #print(top[p])
                        ind_i = top[p][top[p]['item_id'] == i].index.values[0]
                        tmp = top[p].at[ind_i, 'orders']

                        top[p].at[ind_i, 'orders'] = tmp + row['order_count']
                        top[p].at[ind_i, 'order_configs'].append((str(configs), row['order_count']))

                    else:
                        tmp = pd.DataFrame([[i, row['order_count'], [(str(configs), row['order_count'])]]], columns = ['item_id', 'orders', 'order_configs'])
                        top[p] = top[p].append(tmp, ignore_index = True)

                else:
                    break

            configs += 1

            # break the loop if there are no more slots
            if len(top[p].index) >= pf[p]:
                break

        print('')
        while len(top[p].index) < pf[p]:
            
            if (backup[0][0] not in list(top[p].item_id)):
                print(f'incomplete order configuration: {backup[0][0]}')
                tmp = pd.DataFrame([backup[0].append((str(configs), 0))], columns = ['item_id', 'orders', 'order_configs'])
                top[p] = top[p].append(tmp, ignore_index = True)
            backup.pop(0)

            if len(backup) == 0:
                break

        # Calculate the percent of item velocity within the pickface
        sum = top[p].orders.sum()
        top[p]['percent'] = top[p].apply(lambda row: row.orders / sum, axis = 1)

        top[p] = top[p].sort_values('orders', ascending = False).set_index('item_id')
        print(f'\nTop {len(top[p])} Items:\n{top[p]}')
        #print(top[p])
        #print(top[p]['percent'].sum())
        #print(order_count)

        visited = list(order_count[order_count.visited == True].index)
        #print(visited)
        
        ord_serv = order_count[order_count.visited == True].order_count.sum()
        print(f'\nTotal Orders: {ord_sum}')
        print(f'Orders Served by PF: {ord_serv}')
        ord_per = ord_serv / ord_sum
        print('% Orders Served: {:.2%}'.format(ord_per))

        sub_hashkey = hashkey[hashkey.order_config.isin(visited)]
        #print(sub_hashkey)
        #min_max = min_max_from_hashkey(sub_hashkey, case_info)
        #print(min_max)
        
        # Remove all the used order configurations
        order_count = order_count[order_count.visited != True]

        # Save the pickface info to a csv
        #top[p].to_csv(f'{cust}-{pf[p]}_slots.csv')

        pickf = switch_pf(pf[p], cust, 1, 15)
        #print(top[p])
        pickf.populate(top[p])
        pickf.display()
        pickf.to_csv()
        pickfaces.append(pickf)

    return pickfaces
       

