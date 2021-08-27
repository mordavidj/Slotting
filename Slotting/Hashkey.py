import pandas as pd
import pyodbc
pd.set_option('max_columns', None)
import numpy as np
import datetime as dt
from Pickface import *
from DB import connect_db



MIN_LOL_ORDERS = 50
MAX_LOL_LINES = 3
MAX_LOL_ITEMS = 5


def upload_orders(filepath, warehouse, client):
    print(f'\nLoading hashkey: {filepath} . . . ', end = '')

    # Get the file type and read it in appropriately
    f_type = filepath.split('.')[-1]
    if f_type == 'csv':
        df = pd.read_csv(filepath,
                            dtype = 'string')

    elif f_type == 'xlsx':
        df = pd.read_excel(filepath,
                            dtype = 'string')

    elif f_type == 'txt':
        df = pd.read_csv(filepath,
                            dtype = 'string',
                            sep = '^')

    else:
        raise f"Unrecognized filetype: .{f_type} \nExpected filetypes: .txt, .csv, .xlsx"

    print(df)
    print(df.dtypes)

    if 'ORDERNUMBER' in df.columns:
        df = generate_hashkey_ASC(df, client)

    elif 'Optimization Hash Key' in df.columns:
        df = df.rename(columns = {'Created Date': 'date',
                                  'Optimization Hash Key': 'hashkey',
                                  'Client Order Number': 'order_number'})\
                                    [['date', 'order_number', 'hashkey']]
        df['hashkey'] = df['hashkey'].str.replace(r';$', '')

        # Order configurations are hashkeys without quantities
        print('\nGenerating Order Configurations . . . ', end = '')
        df['order_config'] = df.apply(
            lambda row: ';'.join(
                [h.split('*')[0] for h in row['hashkey'].split(';')]), 
            axis =  1)
        
        df['date'] = df.apply(lambda row: pd.to_datetime(row['date']).round(freq='S'), axis = 1)

    else:
        df['date'] = pd.to_datetime(df['date'])
        print(df)

    df['order_number'] = df.order_number.str.strip()
    df['hashkey'] = df.hashkey.str.strip()
    df['order_config'] = df.order_config.str.strip()
    
    df = df.sort_values(by='date', ascending=False)\
        .drop_duplicates(subset=['order_number']).set_index('order_number')
    print('Done')
    print(df)
    cnxn = connect_db(warehouse)
    crsr = cnxn.cursor()

    if crsr.tables(table=client, tableType='TABLE').fetchone():
        sql = 'SELECT TOP 1 [order_date] FROM [{0}] ORDER BY [order_date] DESC'.format(client)
        print(sql)
        crsr.execute(sql)
        res = crsr.fetchone()
        overlap_date = res[0] + dt.timedelta(days = 1)

    else:
        print(f'Creating new table {client} in {warehouse} orders database . . . ')
        crsr.execute('''CREATE TABLE [{0}]
                        ( [order_number]  VARCHAR
                        , [order_date]    DATETIME
                        , [hashkey]       LONGTEXT
                        , [order_config]  LONGTEXT
                        , CONSTRAINT [PrimaryKey] PRIMARY KEY ([order_number])
                        , CONSTRAINT [UniqueKey] UNIQUE ([order_number]));'''.format(client))
        cnxn.commit()
        overlap_date = dt.datetime(1900,1,1,0,0,0)

    new = df[df['date'] > overlap_date]
    old = df[df['date'] <= overlap_date]

    print('Inserting orders into {0} . . . '.format(client))
    print(f'New Orders: {len(new):,}')
    print(f'Old Orders: {len(old):,}')
    if not new.empty:
        sql = f"INSERT INTO [{client}] ([order_number], [order_date], [hashkey], [order_config]) VALUES(Trim([?]), Trim([?]), Trim([?]), Trim([?]))"
        try:
            crsr.executemany(sql, new.itertuples(index=True))
            cnxn.commit()
            print(f'New orders inserted: {len(new):,}')
        except pyodbc.Error as err:
            print(err)
            print(f'Defaulting to serial insertion of {len(new):,} records . . . ')
            old = df
    
    if not old.empty:
        count = 0
        for ind, row in old.iterrows():
            try:
                crsr.execute(f'INSERT INTO [{client}] ([order_number], [order_date], [hashkey], [order_config]) VALUES([?], [?], [?], [?]);',
                             (ind, row['date'], row['hashkey'], row['order_config']))
                count += 1
            except pyodbc.Error as err:
                if err.args[0] != '23000':
                    print(err)

        if count > 0:
            cnxn.commit()
        print(f'{count:,}/{len(old):,} records inserted individually.')

    crsr.close()
    cnxn.close()

    print('Done')



def get_hashkey(warehouse:str, client:str, period:int = 42):
    if warehouse is None or warehouse is '':
        raise Exception("Warehouse must be specified.")
    if client is None or client is '':
        raise Exception("Client must be specified.")

    cnxn = connect_db(warehouse)
    crsr = cnxn.cursor()

    crsr.execute('SELECT TOP 1 order_date FROM {0} ORDER BY order_date DESC'.format(client))
    res = crsr.fetchone()
    latest_date = res[0]
    
    d = (latest_date - dt.timedelta(days=period)).strftime("%Y-%m-%d")

    print('Getting orders from {0} to {1} . . . '.format(
        d, latest_date.strftime("%Y-%m-%d")), end = '')

    orders_sql = '''SELECT t.order_number, t.order_date, t.hashkey, t.order_config
                    FROM {0} AS t
                    WHERE t.order_date >= #{1}#;'''.format(client, d)

    df = pd.read_sql(orders_sql, cnxn).set_index('order_number')

    return df.rename(columns = {'order_date': 'date'})



def load_hashkey(filepath:str = None, warehouse:str = 'WJ', client:str = None, period:int = 42):
    '''Load a Power BI Hashkey report from a file and generate order configurations

    '''
    if client is None:
        raise "Client must be specified."

    df = None

    cnxn = connect_db(warehouse)
    crsr = cnxn.cursor()

    if not pd.isna(filepath) and filepath is not None:
        print(f'\nLoading hashkey: {filepath} . . . ', end = '')

        # Get the file type and read it in appropriately
        f_type = filepath.split('.')[-1]
        if f_type == 'csv':
            df = pd.read_csv(filepath,
                             dtype = 'string')

        elif f_type == 'xlsx':
            df = pd.read_excel(filepath,
                               dtype = 'string')

        elif f_type == 'txt':
            df = pd.read_csv(filepath,
                             dtype = 'string',
                             sep = None,
                             engine = 'python')

        else:
            raise f"Unrecognized filetype: .{f_type} \nExpected filetypes: .txt, .csv, .xlsx"

        if 'Optimization Hash Key' in df.columns:
            df = df.rename(columns = {'Created Date': 'date',
                                 'Optimization Hash Key': 'hashkey',
                                 'Client Order Number': 'order_number'})\
                                     [['date', 'order_number', 'hashkey']]\
                                     .set_index('order_number')
            df['hashkey'] = df['hashkey'].str.replace(r';$', '')

            # Order configurations are hashkeys without quantities
            print('\nGenerating Order Configurations . . . ', end = '')
            df['order_config'] = df.apply(
                lambda row: ';'.join(
                    [h.split('*')[0] for h in row['hashkey'].split(';')]), 
                axis =  1)
            print('Done')
        df['date'] = pd.to_datetime(df['date'])

        crsr.execute('SELECT TOP 1 order_date FROM {0} ORDER BY order_date DESC'.format(client))
        res = crsr.fetchone()
        if res:
            overlap_date = res[0]
            print(len(df))
            print(overlap_date)
            df = df[df['date'] > overlap_date]
            print(len(df))
            print(df)
        print('Inserting orders into db: {0} . . . '.format(warehouse), end = '')
        sql = f"INSERT INTO {client.upper()} (order_number, order_date, hashkey, order_config) VALUES(?, ?, ?, ?)"
        try:
            crsr.executemany(sql, df.itertuples(index=True))
            cnxn.commit()
        except pyodbc.Error as err:
            print(err)
            print('Saving hashkey to excel')
            f_type = 'xlsx'
            path = '.'.join(filepath.split('.')[:-1]) + '.' + f_type
            print(path)
            df.to_excel(path) 

        print('Done')

    crsr.execute('SELECT TOP 1 order_date FROM {0} ORDER BY order_date DESC'.format(client))
    res = crsr.fetchone()
    latest_date = res[0]
    
    d = (latest_date - dt.timedelta(days=period)).strftime("%Y-%m-%d")
    print('Getting orders from {0} to {1} . . . '.format(
        d, latest_date.strftime("%Y-%m-%d")), end = '')
    #print(d)
    orders_sql = '''SELECT t.order_number, t.order_date, t.hashkey, t.order_config
                    FROM {0} AS t
                    WHERE t.order_date >= #{1}#;'''.format(client, d)

    #print(orders_sql)

    df = pd.read_sql(orders_sql, cnxn).set_index('order_number')
    #print(df)
    df = df.rename(columns = {'order_date': 'date'})
    #df['date'] = pd.to_datetime(df['date']).dt.date
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



def generate_hashkey_ASC(df:pd.DataFrame, client:str):
    '''Take a file from ASC and generate hashkeys and order configs

    '''
   
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
                  WHERE i.customer = ucase('{0:s}');'''.format(client)

    items = pd.read_sql(item_sql, cnxn).set_index('ASC_id')
    item_id = list(items.index.values)
    
    
    kit_sql = '''SELECT k.ASC_id
                 FROM Kit AS k
                 WHERE k.customer = ucase('{0:s}');'''.format(client)

    #kits = pd.read_sql(kit_sql, cnxn).set_index('kit_id')

    #kit_id = list(kits.index.values)
    
    crsr.close()
    cnxn.close()

    print('Done\nBuilding hashkey from items and kits . . . ', end = '')
    if 'SHIPDATE' in df.columns:
        col = 'SHIPDATE'
    else:
        col = 'OURORDERDATE'

    for index, row in df.iterrows():
        
        # If the it item can be converted into a date, it's a new item
        try:
            date_time = dt.datetime.strptime(row[col], 
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
                    dict[row['ORDERNUMBER']] += row[col]

                else:
                    dict[row['ORDERNUMBER']] = row[col]

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
    try:
        df['qty'] = df['qty'].astype(float)
    except:
        print(df['qty'].max())
    #print(df)

    # Get the total shipped each day
    df = df.groupby(['date', 'item_id']).agg({'qty': 'sum'})
    df.reset_index(inplace=True)
    #print(df)

    # Pivot so that day sums of all item quantities are aligned on the same row with the date
    pivot_df = df.pivot(index = 'date', columns = 'item_id', values = 'qty')
    min_max = pd.DataFrame(columns = ['item_id', '80', '90'])
    pivot_df = pivot_df.fillna(0)
    #print(pivot_df)

    for i in range(0, len(pivot_df.columns)):
        min_max = min_max.append({'item_id': str(pivot_df.columns[i]),
                                  '80': np.nanpercentile(pivot_df[pivot_df.columns[i]], 80), 
                                  '90': np.nanpercentile(pivot_df[pivot_df.columns[i]], 90)}, 
                                 ignore_index = True)

    #print(min_max)
    tmp = min_max.set_index('item_id')
    tmp['80'] = tmp['80'].astype(int)
    tmp['90'] = tmp['90'].astype(int)
    #print(tmp)
    tmp.to_csv('MANSCAPED_MIN-MAX.csv')

    case_info = case_info[['case_qty']]
    min_max = min_max.set_index('item_id').join(case_info, how='left')

    #print(min_max)
    min_max['min'] = min_max.apply(lambda row: to_int(np.ceil(row['80'] / row['case_qty'])) if row['case_qty'] else 1, axis = 1)
    min_max['max'] = min_max.apply(lambda row: to_int(np.ceil(row['90'] / row['case_qty'])) if row['case_qty'] else 1, axis = 1)
    
    print('Done')
    #print(min_max)

    return min_max[['min', 'max']]



def remove_lol(hashkey: pd.DataFrame):
    '''Filters out any orders in the hashkey that could be handled in LOL

    '''
    
    print("\nRemoving LOL orders . . . ", end = "")
        
    #lol_hashkey = copy.deepcopy(hashkey)
    hashkey['d'] = pd.to_datetime(
        hashkey.apply(lambda row: pd.to_datetime(row['date']) + dt.timedelta(days=1) if row['date'].hour >= 12 else row['date'], 
                          axis = 1))\
                              .dt.date
    
    hashkey['s'] = hashkey.apply(lambda row: 1 if row['date'].hour >= 12 or row['date'].hour < 5 else 2, axis = 1)
    
    by_schedule = hashkey.groupby(['d', 's'])['hashkey'].value_counts().to_frame()
    print(by_schedule)
    
    ord_sum = len(hashkey)

    for ind, row in by_schedule.iterrows():
        if row['hashkey'] >= MIN_LOL_ORDERS:
            hash = ind[2]
            sum = 0
            tmp_items = []
            s_hash = hash.split(';')

            if len(s_hash) <= MAX_LOL_LINES:
                for h in s_hash:
                    item, quant = h.split('*')
                    sum += int(quant)

                if sum <= MAX_LOL_ITEMS:
                    hashkey = hashkey.drop(
                        hashkey[(hashkey['d'] == ind[0])\
                            & (hashkey.s == ind[1])\
                            & (hashkey.hashkey == hash)].index)
 
                        

    #print(hashkey.groupby('date')['hashkey'].value_counts().to_frame())
    print('Done')
    print('\nTotal Orders: {0:,}'.format(ord_sum))
    print('Orders Removed: {0:,} ({1:.2%})'.format(
        ord_sum - len(hashkey), (ord_sum - len(hashkey)) / ord_sum))

    print('Remaining: {0:,} ({1:.2%})'.format(len(hashkey), len(hashkey) / ord_sum))
    #print(hashkey)
    return hashkey.drop(columns=['d','s'])



def to_int(val):
    '''Format NaN's, None's, and others into integers
    
    '''
    if math.isnan(val) or val is None or pd.isna(val):
        return 0
    else:
        return int(val)
