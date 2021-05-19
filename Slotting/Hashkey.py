import pandas as pd
pd.set_option('max_columns', None)
import numpy as np
import datetime
from Pickface import *
from DB import connect_db


def load_powerBI_hashkey(filepath):
    '''Load a Power BI Hashkey report from a file and generate order configurations

    '''
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



def load_hashkey(filepath):
    '''Load a hashkey that has already been generated.

    '''
    print(f'\nLoading Hashkey: {filepath} . . . ', end = '')

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

    item_sql = '''SELECT i.item_id, i.status
                  FROM Item AS i
                  INNER JOIN Customer AS c ON i.customer = c.customer_id
                  WHERE c.customer = ucase('{0:s}');'''.format(cust)

    items = pd.read_sql(item_sql, cnxn).set_index('item_id')
    item_id = list(items.index.values)
    

    kit_sql = '''SELECT k.kit_id, k.status
                 FROM Kit AS k
                 INNER JOIN Customer AS c ON k.customer = c.customer_id
                 WHERE c.customer = ucase('{0:s}');'''.format(cust)

    kits = pd.read_sql(kit_sql, cnxn).set_index('kit_id')

    kit_id = list(kits.index.values)
    
    print('Done\nBuilding hashkey from items and kits . . . ', end = '')

    for index, row in df.iterrows():
        
        # If the it item can be converted into a date, it's a new item
        try:
            date_time = datetime.datetime.strptime(row['SHIPDATE'], 
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
                if items.loc[row['ORDERNUMBER'], 'status'] != 'O':
                    if row['ORDERNUMBER'] in dict.keys():
                        dict[row['ORDERNUMBER']] += row['SHIPDATE']

                    else:
                        dict[row['ORDERNUMBER']] = row['SHIPDATE']

            elif row['ORDERNUMBER'] in kit_id:
                continue


    n_row.append(str)
    #print(tmp)
    n_frame.append(n_row.copy())

    # first item is a blank line, so pop it
    n_frame.pop(0)
    hashkey = pd.DataFrame(n_frame, columns = ['date', 'order_number', 
                                               'hashkey', 'order_config'])

    hashkey = hashkey[hashkey['hashkey'] != '']
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
    #print(df)
    
    # Split the hashkey into items and quantities
    df[['item_id', 'qty']] = df['hashkey'].str.split('*', expand=True)
    df['qty'] = df['qty'].astype(int)
    #print(df)

    # Get the total shipped each day
    df = df.groupby(['date', 'item_id']).agg({'qty': 'sum'})
    df.reset_index(inplace=True)
    #print(df)

    # Pivot so that day sums of all item quantities are aligned on the same row with the date
    pivot_df = df.pivot(index = 'date', columns = 'item_id', values = 'qty')
    min_max = pd.DataFrame(columns = ['item_id', '80', '90'])
    #print(pivot_df)

    for i in range(0, len(pivot_df.columns)):
        min_max = min_max.append({'item_id': str(pivot_df.columns[i]),
                                  '80': np.nanpercentile(pivot_df[pivot_df.columns[i]], 80), 
                                  '90': np.nanpercentile(pivot_df[pivot_df.columns[i]], 90)}, 
                                 ignore_index = True)

    #print(min_max)
    case_info = case_info[['case_qty']]
    min_max = min_max.set_index('item_id').join(case_info, how='left')

    min_max['min'] = min_max.apply(lambda row: to_int(np.ceil(row['80'] / row['case_qty'])), axis = 1)
    min_max['max'] = min_max.apply(lambda row: to_int(np.ceil(row['90'] / row['case_qty'])), axis = 1)
    
    print('Done')
    #print(min_max)

    return min_max[['min', 'max']]


def to_int(val):
    '''Format NaN's, None's, and others into integers
    
    '''
    if math.isnan(val) or val is None or pd.isna(val):
        return ''
    else:
        return int(val)