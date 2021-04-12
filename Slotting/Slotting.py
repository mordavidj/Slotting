import pandas as pd
import numpy as np
import datetime
from Pickface import *

def load_powerBI_hashkey(filepath):
    print(f'Loading Power BI Hashkey: {filepath}')

    df = pd.read_csv(filepath,
                     dtype = 'string')\
                         .rename(columns = {'Created Date': 'date',
                                            'Optimization Hash Key': 'hashkey'})
    df['date'] = pd.to_datetime(df['date']).dt.date
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
    print(df.head(20))
    print(df.dtypes)
    return df

def load_hashkey(filepath):

    df = pd.read_csv(filepath,
                     dtype = 'string')

    df['date'] = pd.to_datetime(df['date']).dt.date

    
    return df

#################################################
# Take Order numbers, items, and quantities to build a hash key
#################################################
def generate_hashkey(df):
    df['hashkey'] = df.apply(lambda row: row.item_id + '*' + str(row.quantity), axis = 1)
    #print(df.groupby('order_number')['hashkey'].apply(lambda row: ';'.join(row.part_number + '*' + row.Quantity)).reset_index())
    df = df.groupby(['order_number'])['hashkey'].apply(lambda row: ';'.join(row)).reset_index()
    #print(df.head())
    return df

def generate_hashkey_ASC(filepath):

    print(f'\nGenerating Hashkey from ASC file: {filepath}', end = '')

    hash = ''
    config = ''
    prefix = ''
    n_frame = []
    n_row = []

    for index, row in df.iterrows():
        
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
            hash += prefix + f'{row["ORDERNUMBER"]}*{row["SHIPDATE"]}'
            config += prefix + str(row["ORDERNUMBER"])
            prefix = ';'

    n_row.append(str)
    tmp = n_row
    #print(tmp)
    n_frame.append(n_row.copy())
    n_frame.pop(0)
    hashkey = pd.DataFrame(n_frame, columns = ['date', 'order_number', 'hashkey', 'order_config'])

    print('Done')
    return hashkey


def generate_hashkey_SQL(filepath):
    df = None

    f_type = filepath.split('.')[-1]
    if f_type == 'csv':
        df = pd.read_csv(filepath,
                         dtype = 'string')

    elif f_type == 'xlsx':
        df = pd.read_excel(filepath,
                           dtype = 'string',
                           header = 1)\
                               .rename(columns = {'Ship Date': 'date',
                                                  'Order Number': 'order_number',
                                                  'Item ID': 'item_id',
                                                  'Qty\nShipped': 'qty'})

    print(df.head())

    df['qty'] = df.apply(lambda row: '1' if row.qty == 0 else row.qty, axis = 1)

    df['date'] = pd.to_datetime(df['date']).dt.date
  
    df['hashkey'] = df.apply(lambda row: row.item_id + '*' + row.qty, axis = 1)
    
    hashkey = df.groupby(['order_number', 'date'])['hashkey'].apply(lambda row: ';'.join(row)).reset_index()

    print(hashkey.head())
    return hashkey


def min_max_from_hashkey(hashkey, case_info):
    
    df = pd.concat([pd.Series(row['date'], row['hashkey'].split(';')) for _, row in hashkey.iterrows()]).reset_index()
    df = df.rename(columns = {'index': 'hashkey', 0: 'date'})
    df = df[df['hashkey'] != '']
    #print(df.head())
    
    df[['item_id', 'qty']] = df['hashkey'].str.split('*', expand=True)
    df['qty'] = df['qty'].astype(int)
    #print(df.head())

    df = df.groupby(['date', 'item_id']).agg({'qty': 'sum'})
    df.reset_index(inplace=True)
    #print(df.head())

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


def to_int(val):
    if math.isnan(val) or val is None:
        return ''
    else:
        return int(val)


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

def continuous_slotting(hashkey, pf, cust):
    if type(pf) is not list:
        pf = [pf]

    for p in pf:
        assert (type(p) == int or type(p) == float) and p > 0

    assert cust != ''
    
    #print(hashkey)
    order_count = hashkey.order_config.value_counts().to_frame()\
        .rename(columns={'order_config': 'order_count'})
    order_count['visited'] = False

    item_ids = set()
    for index, row in order_count.iterrows():
        for i in index.split(';'):
            item_ids.add(i)
    items_df = pd.DataFrame(item_ids, columns = ['item_id']).sort_values('item_id').set_index('item_id')

    items_df['used'] = False
    print(items_df)

    print(order_count)

    #case_info = pd.read_excel('../../../Documents/Master Case info.xlsx',
    #                     sheet_name = 'Case')[['item_id', 'case_qty']]

    top = []
    pickfaces = []
    ord_sum = order_count.order_count.sum()
    backup = []
    configs = 0

    # loop through each pickface number and get those top X items
    for p in range(len(pf)):
        
        # each pickface is stored in its own dataframe
        top.append(pd.DataFrame(columns = ['item_id', 'orders', 'order_configs']))
        if backup:
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

        pickface = switch_pf(pf[p], cust, 1, 15)
        #print(top[p])
        pickface.load(top[p])
        pickface.display()
        pickface.to_csv()
        pickfaces.append(pickface)

    return pickfaces

def slotting(hashkey, pf, cust):

    if type(pf) is not list:
        pf = [pf]

    for p in pf:
        assert (type(p) == int or type(p) == float) and p > 0

    assert cust != ''
    
    #print(hashkey)
    order_count = hashkey.order_config.value_counts().to_frame()\
        .rename(columns={'order_config': 'order_count'})
    order_count['visited'] = False

    hashkey_count = hashkey.hashkey.value_counts().to_frame()
    print(order_count)

    #case_info = pd.read_excel('../../../Documents/Master Case info.xlsx',
    #                     sheet_name = 'Case')[['item_id', 'case_qty']]

    top = []
    pickfaces = []
    ord_sum = order_count.order_count.sum()
    
    # loop through each pickface number and get those top X items
    for p in range(len(pf)):
        configs = 0
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

        pickface = switch_pf(pf[p], cust, 1, 15)
        #print(top[p])
        pickface.load(top[p])
        pickface.display()
        pickface.to_csv()
        pickfaces.append(pickface)

    return pickfaces
       

